# Written by Martin Fajcik <martin.fajcik@vut.cz>
#
import copy
import json
import logging
import multiprocessing
import os
import pickle
import random
import re
import time
from collections import defaultdict
from functools import partial
from math import ceil
from multiprocessing import Pool
from random import shuffle
from typing import List, AnyStr, Optional, Union

import torch
import torch.distributed as dist
from jsonlines import jsonlines
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, RobertaTokenizerFast, RobertaTokenizer

from ..tokenizer.init_tokenizer import init_tokenizer
from ....common.db import PassageDB
from ....common.utility import count_lines, unicode_normalize, deduplicate_list, mkdir, get_random_context_size

logger = logging.getLogger(__name__)


def process_evid(sentence):
    # Normalization taken from Dominik Stammbach
    sentence = unicode_normalize(sentence)
    sentence = re.sub(" -LSB-.*?-RSB-", " ", sentence)
    sentence = re.sub(" -LRB- -RRB- ", " ", sentence)
    sentence = re.sub("-LRB-", "(", sentence)
    sentence = re.sub("-RRB-", ")", sentence)
    sentence = re.sub("-COLON-", ":", sentence)
    sentence = re.sub("--", "-", sentence)
    sentence = re.sub("``", '"', sentence)
    sentence = re.sub("''", '"', sentence)
    return sentence


def has_sent_hit(example, topk_titles_raw):
    if not example['metadata']['evidence'] or example['label'] == 'NOT ENOUGH INFO':
        return False
    pred_id_sent_list = [[int(x) for x in y.split("|")] for y in example["metadata"]['sent_ranges']]
    # for every annotation
    for annotated_group in example['metadata']['evidence']:
        sent_hits = []
        # for every annotated item in single annotation
        for _, _, doc_title, sent_id in annotated_group:
            sent_hit = False
            for idx, topk_title in enumerate(topk_titles_raw):
                if topk_title == doc_title:
                    if sent_id in pred_id_sent_list[idx]:
                        sent_hit = True
                        break
            sent_hits.append(sent_hit)

        if all(sent_hits):  # there can be many GTs, but hitting at least 5 is OK
            return True
    return False


def is_impossible_fever_score(example):
    return all(len(annotated_group) > 5 for annotated_group in example['metadata']['evidence'])


def preprocessing_method_parallel_wrapper(preprocessing_method, example):
    global kwargs
    return preprocessing_method(example, **kwargs)


def init(_kwargs):
    global kwargs
    _kwargs['database'] = PassageDB(_kwargs['database'])
    _kwargs['tokenizer'] = init_tokenizer(_kwargs['tokenizer'])
    # Disable tokenization errors
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    kwargs = _kwargs


class FEVERLRMVerifierDataset(IterableDataset):
    def __init__(self, data_file: AnyStr, tokenizer: PreTrainedTokenizer,
                 transformer, database, context_length, block_size, max_len=None,
                 is_training=True, include_golden_passages=True, shuffle=False, skip_NEI=False, jiangetal_sup=None,
                 expand_ret=None, extra_context_size=None,
                 eval_interpretability=False, cheat_on_val=False, randomize_context_lengths=False,
                 cache_dir='.data/lrm_verifier', distributed_settings=None):
        self.cache_dir = cache_dir
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.database = database if not type(database) == dict else database["fever"]
        self.transformer = transformer
        self.max_len = max_len
        self.cheat_on_val = cheat_on_val
        self.block_size = block_size
        self.is_training = is_training
        self.context_length = context_length
        self.shuffle = shuffle
        self.skip_NEI = skip_NEI
        self.include_golden_passages = include_golden_passages
        self.distributed_settings = distributed_settings
        self.retrieve_whole_articles = True # other options are not supported anymore
        self.randomize_context_lengths = randomize_context_lengths
        self.jiangetal_sup = jiangetal_sup
        self.expand_ret = expand_ret
        self.extra_context_size = extra_context_size

        self.eval_interpretability = eval_interpretability
        if eval_interpretability:
            with jsonlines.open(eval_interpretability) as r:
                annotated_dataset = list(r)
            self.data_subset = [e['id'] for e in annotated_dataset]

        preprocessed_f = self.create_preprocessed_name()
        self.preprocessed_f = preprocessed_f
        if not os.path.exists(preprocessed_f):
            logger.info(f"{preprocessed_f} not found!\nCreating new preprocessed file...")
            if distributed_settings is not None:
                if dist.get_rank() == 0:
                    self.preprocess_data(preprocessed_f)
                dist.barrier()  # wait for preprocessing to be finished by 0
            else:
                self.preprocess_data(preprocessed_f)

        self.index_dataset()
        self._total_data = len(self._line_offsets)
        if distributed_settings is not None:
            standard_part_size = ceil(self._total_data / distributed_settings["world_size"])
            self._total_data_per_rank = standard_part_size \
                if (distributed_settings['rank'] < distributed_settings["world_size"] - 1 or is_training
                    ) else \
                self._total_data - (distributed_settings["world_size"] - 1) * standard_part_size

    def preprocess_data(self, preprocessed_f):
        s_time = time.time()
        self._preprocess_data()
        logger.info(f"Dataset {preprocessed_f} created in {time.time() - s_time:.2f}s")

    def __len__(self):
        return self._total_data if not dist.is_initialized() else self._total_data_per_rank

    def get_example(self, n: int) -> str:
        """
        Get n-th line from dataset file.
        :param n: Number of line you want to read.
        :type n: int
        :return: the line
        :rtype: str
        Author: Martin Docekal, modified by Martin Fajcik
        """
        if self.preprocessed_f_handle.closed:
            self.preprocessed_f_handle = open(self.preprocessed_f)

        self.preprocessed_f_handle.seek(self._line_offsets[n])
        return json.loads(self.preprocessed_f_handle.readline().strip())

    def index_dataset(self):
        """
        Makes index of dataset. Which means that it finds offsets of the samples lines.
        Author: Martin Docekal, modified by Martin Fajcik
        """

        lo_cache = self.preprocessed_f + "locache.pkl"
        if os.path.exists(lo_cache):
            logger.info(f"Using cached line offsets from {lo_cache}")
            with open(lo_cache, "rb") as f:
                self._line_offsets = pickle.load(f)
        else:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    logger.info(f"Getting lines offsets in {self.preprocessed_f}")
                    self._index_dataset(lo_cache)
                dist.barrier()  # wait for precaching to be finished by 0
                if dist.get_rank() > 0:
                    logger.info(f"Using cached line offsets from {lo_cache}")
                    with open(lo_cache, "rb") as f:
                        self._line_offsets = pickle.load(f)
            else:
                logger.info(f"Getting lines offsets in {self.preprocessed_f}")
                self._index_dataset(lo_cache)

    def _index_dataset(self, lo_cache):
        self._line_offsets = [0]
        with open(self.preprocessed_f, "rb") as f:
            while f.readline():
                self._line_offsets.append(f.tell())
        del self._line_offsets[-1]
        # cache file index
        with open(lo_cache, "wb") as f:
            pickle.dump(self._line_offsets, f)

    def __iter__(self):
        self.preprocessed_f_handle = open(self.preprocessed_f)
        self.order = list(range(self._total_data))
        if self.shuffle:
            logger.info("Shuffling file index...")
            shuffle(self.order)
        if dist.is_initialized():
            distributed_shard_size = ceil(self._total_data / self.distributed_settings["world_size"])
            self.shard_order = self.order[self.distributed_settings["rank"] * distributed_shard_size:
                                          (self.distributed_settings["rank"] + 1) * distributed_shard_size]
            if len(self.shard_order) < distributed_shard_size and self.is_training:
                logger.info(
                    f"Padding process {os.getpid()} with rank {self.distributed_settings['rank']} with "
                    f"{distributed_shard_size - len(self.shard_order)} samples")
                self.padded = distributed_shard_size - len(self.shard_order)
                self.shard_order += self.order[:distributed_shard_size - len(self.shard_order)]
            self.order = self.shard_order
        self.offset = 0
        return self

    def __next__(self):
        if self.offset >= len(self.order):
            if not self.preprocessed_f_handle.closed:
                self.preprocessed_f_handle.close()
            raise StopIteration
        example = self.get_example(self.order[self.offset])
        self.offset += 1
        return example

    def create_preprocessed_name(self):
        transformer = self.transformer.replace('/', '_')
        RCS = 'RCS' if self.randomize_context_lengths else ''
        without_psg_suffix = f"_withoutpassages" if not self.include_golden_passages else ""
        maxlen = f'_L{self.max_len}' if self.max_len is not None else ''
        mkdir(self.cache_dir)
        preprocessed_f_noext = os.path.join(self.cache_dir, os.path.basename(
            self.data_file)) + f"_verifier_preprocessed_for" \
                               f"_{transformer}" \
                               f"{'_wholearticles' if self.retrieve_whole_articles else ''}" \
                               f"{'_noNEI' if self.skip_NEI else ''}" \
                               f"_C{self.context_length}{RCS}" \
                               f"_B{self.block_size}" \
                               f"{without_psg_suffix}" \
                               f"{'_cheat_on_val' if self.cheat_on_val else ''}" \
                               f"{'_interp' if self.eval_interpretability else ''}" \
                               f"{'_jiangSUP' if self.jiangetal_sup is not None else ''}" \
                               f"{'_expand_entities' + str(self.extra_context_size) if self.expand_ret is not None else ''}" \
                               f"{'' if self.expand_ret is not None else ''}" \
                               f"{maxlen}"
        preprocessed_f = preprocessed_f_noext + ".jsonl"
        return preprocessed_f

    # @staticmethod
    # def extract_example(e):
    #     _preprocessed_example = [
    #         e["label"],
    #         e["sources"],
    #         [[1] * len(x) for x in e["sources"]],
    #         e["metadata"]]
    #     """
    #     Prepare example
    #         @staticmethod
    #     def prepare_fields(pad_t):
    #     WORD_field = NestedField(Field(use_vocab=False, batch_first=True, sequential=True, pad_token=pad_t))
    #     PAD_field = NestedField(Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0))
    #
    #     fields = {
    #         'label': RawField(),
    #         'src': WORD_field,
    #         'src_mask': PAD_field,
    #         'metadata': RawField(),
    #     }
    #     return fields
    #     example = Example.fromlist(_preprocessed_example, fields)
    #     """
    #     return _preprocessed_example

    @staticmethod
    def create_collate_fn(pad_t):
        def pad_sequences(sequences, padding_value):
            sequence_lengths = [len(seq) for seq in sequences]
            max_length = max(sequence_lengths)
            padded_sequences = [seq + [padding_value] * (max_length - len(seq)) for seq in sequences]
            return padded_sequences

        def collate_fn(batch):
            label_list, sources_list, source_type_ids_list, source_masks_list, metadata_list = [], [], [], [], []
            for e in batch:
                if 'label' in e:
                    label_list.append(e["label"])
                sources_list.append(pad_sequences(e["sources"], padding_value=pad_t))
                if e["source_type_ids"]:
                    source_type_ids_list.append(pad_sequences(e["source_type_ids"], padding_value=0))
                source_masks_list.append(
                    pad_sequences([[1] * len(x) for x in e["sources"]], padding_value=0))
                metadata_list.append(e["metadata"])

            if len(source_type_ids_list) > 0:
                source_type_ids_list = torch.LongTensor(source_type_ids_list)
            else:
                source_type_ids_list = None

            return torch.LongTensor(sources_list), \
                source_type_ids_list, \
                torch.LongTensor(source_masks_list), \
                label_list, \
                metadata_list

        return collate_fn

    def _preprocess_data(self):
        num_lines = count_lines(self.data_file)
        number_of_short_contexts = 0
        accumulated_context_length = 0
        hits_at_K = 0
        mh_hits_at_K = 0
        mh_art_hits_at_K = 0

        total_enoughinfo_samples = 0

        total_samples = 0
        total_multihop_samples = 0
        total_multihop_art_samples = 0

        total_sentences = 0
        total_articles = 0

        jiangetal_positives_negatives = None
        if self.jiangetal_sup is not None:
            cached_data = os.path.join(self.cache_dir, "jiangetal_sup.pkl")
            if os.path.exists(cached_data):
                with open(cached_data, "rb") as f:
                    jiangetal_positives_negatives = pickle.load(f)
            else:
                jiangetal_positives_negatives = defaultdict(lambda: [])
                TEXT_PATTERN = re.compile(r"Query: (.*) Document: (.*) \. (.*) Relevant:\t(true|false)\n")
                with open(self.jiangetal_sup["texts"]) as retrieved_text_fd, open(
                        self.jiangetal_sup["ids"]) as retrieved_id_fd:
                    for text_data, id_data in zip(retrieved_text_fd, retrieved_id_fd):
                        claim_id, doc_sent_id, rank = id_data.strip().split("\t")
                        document_id, sentence_id = doc_sent_id.rsplit("_", 1)
                        claim, title, sentence, relevance = TEXT_PATTERN.match(text_data).groups()
                        assert relevance in ['true', 'false']
                        jiangetal_positives_negatives[int(claim_id)].append(
                            [unicode_normalize(document_id), int(sentence_id), relevance == 'true'])
                with open(cached_data, "wb") as f:
                    pickle.dump(dict(jiangetal_positives_negatives), f)

        num_processes = multiprocessing.cpu_count()
        with jsonlines.open(self.data_file, "r") as fd, jsonlines.open(self.preprocessed_f, "w") as wf:
            buffer = []
            mh_status_buffer = []
            mh_art_status_buffer = []
            impossibles_at_K = 0
            mh_impossibles_at_K = 0
            hyp = None
            extra_context_size = None
            if self.expand_ret is not None:
                with open(self.expand_ret["hyperlinks_per_sentence"], "rb") as f:
                    hyp = pickle.load(f)
                extra_context_size = self.extra_context_size
            kwargs = {"database": self.database.path,
                      "tokenizer": {
                          'verifier_tokenizer_type': self.tokenizer.name_or_path,
                          'transformers_cache': ".Transformers_cache"
                      },
                      "jiang_data": jiangetal_positives_negatives,
                      "max_input_length": self.max_len,
                      "context_size": self.context_length,
                      "include_golden_passages": self.include_golden_passages,
                      "is_training": self.is_training,
                      "cheat_on_val": self.cheat_on_val,
                      "hyperlinks_per_sentence": hyp,
                      "randomize_context_lengths": self.randomize_context_lengths,
                      "extra_context_size": extra_context_size
                      }
            log_first = True
            sentence_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sentence_special_token)

            pbar = tqdm(enumerate(fd), total=num_lines)
            with Pool(processes=num_processes, initializer=init, initargs=[kwargs]) as pool:
                for idx, sample in pbar:
                    is_test_set = not 'label' in sample
                    if not is_test_set:
                        if self.eval_interpretability and sample['id'] not in self.data_subset:
                            continue
                        is_multihop_sample = sample["label"] != "NOT ENOUGH INFO" and all(
                            len(e) > 1 for e in sample['evidence'])
                        is_multihop_article_sample = all(len(set(ex[-2] for ex in e)) > 1 for e in sample['evidence'])
                        if self.skip_NEI and sample["label"] == "NOT ENOUGH INFO":
                            continue
                        if is_multihop_sample:
                            total_multihop_samples += 1
                        if is_multihop_article_sample:
                            total_multihop_art_samples += 1
                        mh_status_buffer.append(is_multihop_sample)
                        mh_art_status_buffer.append(is_multihop_article_sample)
                    buffer.append(sample)
                    if len(buffer) == num_processes:
                        result = pool.map(
                            partial(preprocessing_method_parallel_wrapper,
                                    FEVERLRMVerifierDataset.process_sample_perart),
                            buffer)
                        # kwargs = {  # "sent_mapping": self.sent_mapping,
                        #     "database": self.database,
                        #     "tokenizer": self.tokenizer,
                        #     "max_input_length": self.max_len,
                        #     "context_size": self.context_length,
                        #     "jiang_data": jiangetal_positives_negatives,
                        #     "include_golden_passages": self.include_golden_passages,
                        #     "is_training": self.is_training,
                        #     "hyperlinks_per_sentence": hyp
                        # }
                        # kwargs = {"database": self.database,
                        #           "tokenizer": self.tokenizer,
                        #           "jiang_data": jiangetal_positives_negatives,
                        #           "max_input_length": self.max_len,
                        #           "context_size": self.context_length,
                        #           "include_golden_passages": self.include_golden_passages,
                        #           "is_training": self.is_training,
                        #           "cheat_on_val": self.cheat_on_val,
                        #           "hyperlinks_per_sentence": hyp,
                        #           "randomize_context_lengths": self.randomize_context_lengths,
                        #           "extra_context_size": extra_context_size
                        #           }
                        # result = [FEVERLRMVerifierDataset.process_sample_perart(e, **kwargs) for e in buffer]

                        total_sentences += sum(
                            [s.count(sentence_special_token) for x in result for s in x[0]['sources']])
                        total_samples += len(buffer)
                        accumulated_context_length, \
                            hits, \
                            impossibles, \
                            number_of_short_contexts, \
                            total_enoughinfo_samples, \
                            total_articles = self.write_processed(
                            accumulated_context_length, log_first, number_of_short_contexts, pbar,
                            result, total_enoughinfo_samples, total_articles, wf)
                        if not is_test_set:
                            hits_at_K += sum(hits)
                            mh_hits_at_K += sum(h for h, is_mh in zip(hits, mh_status_buffer) if is_mh)
                            mh_art_hits_at_K += sum(h for h, is_mh in zip(hits, mh_art_status_buffer) if is_mh)
                            impossibles_at_K += sum(impossibles)
                            mh_impossibles_at_K += sum(
                                imp for imp, is_mh in zip(impossibles, mh_status_buffer) if is_mh)

                            if total_multihop_art_samples > 0:
                                pbar.set_description(
                                    f"Sentence-Recall@{self.context_length}: {hits_at_K * 100 / total_enoughinfo_samples:.2f}, "
                                    f"MH_Recall@{self.context_length}: {mh_hits_at_K * 100 / total_multihop_samples:.2f}, "
                                    f"MH-ART_Recall@{self.context_length}: {mh_art_hits_at_K * 100 / total_multihop_art_samples:.2f}")
                            elif total_multihop_samples > 0:
                                pbar.set_description(
                                    f"Sentence-Recall@{self.context_length}: {hits_at_K * 100 / total_enoughinfo_samples:.2f}, "
                                    f"MH_Recall@{self.context_length}: {mh_hits_at_K * 100 / total_multihop_samples:.2f}")
                            elif total_enoughinfo_samples > 0:
                                pbar.set_description(
                                    f"Sentence-Recall@{self.context_length}: {hits_at_K * 100 / total_enoughinfo_samples:.2f}")
                        log_first = False
                        buffer = []
                        mh_status_buffer = []
                        mh_art_status_buffer = []
                if len(buffer) > 0:
                    result = pool.map(
                        partial(preprocessing_method_parallel_wrapper, FEVERLRMVerifierDataset.process_sample_perart),
                        buffer)
                    total_samples += len(buffer)
                    accumulated_context_length, \
                        hits, \
                        impossibles, \
                        number_of_short_contexts, \
                        total_enoughinfo_samples, \
                        total_articles = self.write_processed(
                        accumulated_context_length, log_first, number_of_short_contexts, pbar,
                        result, total_enoughinfo_samples, total_articles, wf)
                    hits_at_K += sum(hits)
                    mh_hits_at_K += sum(h for h, is_mh in zip(hits, mh_status_buffer) if is_mh)
                    mh_art_hits_at_K += sum(h for h, is_mh in zip(hits, mh_art_status_buffer) if is_mh)

                    impossibles_at_K += sum(impossibles)
                    mh_impossibles_at_K += sum(imp for imp, is_mh in zip(impossibles, mh_status_buffer) if is_mh)
                    buffer = []
                    mh_status_buffer = []
                    mh_art_status_buffer = []
        if not is_test_set:
            logger.info(
                f"Sentence hit @ {self.context_length}: {hits_at_K * 100 / total_enoughinfo_samples:.2f}"
                f" ({hits_at_K}/{total_enoughinfo_samples})")
            logger.info(
                f"MH Sentence hit @ {self.context_length}: {mh_hits_at_K * 100 / total_multihop_samples:.2f}"
                f" ({mh_hits_at_K}/{total_multihop_samples})")
            logger.info(
                f"MH-Art Sentence hit @ {self.context_length}: {mh_art_hits_at_K * 100 / total_multihop_art_samples:.2f}"
                f" ({mh_art_hits_at_K}/{total_multihop_art_samples})")
            logger.info(
                f"Impossibles@5 {self.context_length}: {impossibles_at_K * 100 / total_enoughinfo_samples:.2f}"
                f" ({impossibles_at_K}/{total_enoughinfo_samples})")
            logger.info(
                f"MH Impossibles@5 {self.context_length}: {mh_impossibles_at_K * 100 / total_multihop_samples:.2f}"
                f" ({mh_impossibles_at_K}/{total_multihop_samples})")
            logger.info(f"Average # of sentences at input {total_sentences / total_samples:.2f}")
            logger.info(f"Average # of articles at input {total_articles / total_samples:.2f}")
            logger.info(f"Average context_length: {accumulated_context_length / total_samples:.2f}")
            logger.info(f"Number of contexts shorter than context_length {number_of_short_contexts}")

            logger.info(
                f"Total multihop samples: {total_multihop_samples}/{total_samples} "
                f"({total_multihop_samples / total_samples * 100 :.2f}%)")
            logger.info(
                f"Total multihop-art samples: {total_multihop_art_samples}/{total_samples} "
                f"({total_multihop_art_samples / total_samples * 100 :.2f}%)")
        else:
            logger.info(f"Average # of sentences at input {total_sentences / total_samples:.2f}")
            logger.info(f"Average # of articles at input {total_articles / total_samples:.2f}")
            logger.info(f"Average context_length: {accumulated_context_length / total_samples:.2f}")
            logger.info(f"Number of contexts shorter than context_length {number_of_short_contexts}")

    def write_processed(self, accumulated_context_length, log_first, number_of_short_contexts, pbar,
                        result, total_enoughinfo_samples, total_articles, wf):
        hits = []
        impossibles = []
        for example, topk_titles_raw in result:
            is_test_set = 'label' not in example
            if not is_test_set:
                hits.append(has_sent_hit(example, topk_titles_raw))
                impossibles.append(is_impossible_fever_score(example))

                total_articles += len(set(topk_titles_raw))
                if example['label'] != 'NOT ENOUGH INFO':
                    total_enoughinfo_samples += 1
                assert example is not None
                accumulated_context_length += len(example["sources"])
                if len(example["sources"]) < self.context_length:
                    number_of_short_contexts += 1
            wf.write(example)
            if log_first:
                logger.info("Example of model input formats:")

                def log_sample(e):
                    logger.info("*" * 10)
                    src_example1 = " ".join(self.tokenizer.convert_ids_to_tokens(e["sources"][-1]))
                    logger.info("inputs 1:")
                    logger.info(src_example1)

                log_sample(example)
        return accumulated_context_length, hits, impossibles, number_of_short_contexts, total_enoughinfo_samples, total_articles

    @staticmethod
    def assemble_input_sequences(claim: List[int], passages: List[List[int]], tokenizer: PreTrainedTokenizer,
                                 max_input_length: int,
                                 titles: Optional[List[List[int]]] = None):
        inputs = []
        sentence_counts = []
        truncation_flags = []
        input_type_ids = []

        if type(tokenizer) in [RobertaTokenizer, RobertaTokenizerFast]:
            claim_special_token = tokenizer.convert_tokens_to_ids(tokenizer.claim_special_token)
            passage_special_token = tokenizer.convert_tokens_to_ids(tokenizer.passage_special_token)
            title_special_token = tokenizer.convert_tokens_to_ids(tokenizer.title_special_token)
            sentence_special_token = tokenizer.convert_tokens_to_ids(tokenizer.sentence_special_token)
            for title, passage in zip(titles, passages):
                claim_and_title = [tokenizer.cls_token_id] + [claim_special_token] + claim + \
                                  [tokenizer.sep_token_id, tokenizer.sep_token_id] + [title_special_token] + title + [
                                      passage_special_token]
                truncate_flag = False
                seq = claim_and_title + passage
                if len(seq) > max_input_length - 1:
                    if not seq[max_input_length - 3] == sentence_special_token:
                        seq = seq[:max_input_length - 2] + [sentence_special_token]
                    else:
                        seq = seq[:max_input_length - 1]
                    truncate_flag = True

                seq = seq + [tokenizer.eos_token_id]

                sentence_count = seq.count(sentence_special_token)
                assert seq.count(claim_special_token) == 1
                assert sentence_count > 0

                inputs.append(seq)
                sentence_counts.append(sentence_count)
                truncation_flags.append(truncate_flag)
        else:
            claim_special_token = tokenizer.convert_tokens_to_ids(tokenizer.claim_special_token)
            passage_special_token = tokenizer.convert_tokens_to_ids(tokenizer.passage_special_token)
            title_special_token = tokenizer.convert_tokens_to_ids(tokenizer.title_special_token)
            sentence_special_token = tokenizer.convert_tokens_to_ids(tokenizer.sentence_special_token)
            claim = [claim_special_token] + claim

            for title, passage in zip(titles, passages):
                title = [title_special_token] + title
                passage = [passage_special_token] + passage
                truncate_flag = False
                seq = tokenizer.build_inputs_with_special_tokens(claim, title + passage)
                input_type_id = tokenizer.create_token_type_ids_from_sequences(claim, title + passage)[
                                :max_input_length]
                if len(seq) > max_input_length - 1:
                    if not seq[max_input_length - 3] == sentence_special_token:
                        seq = seq[:max_input_length - 2] + [sentence_special_token]
                    else:
                        seq = seq[:max_input_length - 1]
                    truncate_flag = True
                    seq = seq + [
                        tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id]
                sentence_count = seq.count(sentence_special_token)
                assert sentence_count > 0
                assert seq.count(claim_special_token) == 1
                assert len(seq) == len(input_type_id)

                # Make sure there are no empty sentences
                for i in range(len(seq) - 1):
                    assert tuple(seq[i:i + 2]) != (sentence_special_token, sentence_special_token)

                inputs.append(seq)
                sentence_counts.append(sentence_count)
                truncation_flags.append(truncate_flag)
                input_type_ids.append(input_type_id)
        # else:
        #     assert False, "Unsupported tokenizer"

        return inputs, input_type_ids, sentence_counts, truncation_flags

    @staticmethod
    def process_sample_perart(sample: dict,
                              database: Union[PassageDB, AnyStr],
                              tokenizer: PreTrainedTokenizer,
                              context_size: int,
                              include_golden_passages=True,
                              max_input_length: Union[int, None] = None,
                              is_training: bool = False,
                              cheat_on_val: bool = False,
                              randomize_context_lengths: bool = False,
                              B: int = 6,
                              max_gt_articles=10,
                              jiang_data=None,
                              truncate_gts=True,
                              hyperlinks_per_sentence=None,
                              extra_context_size=None):
        """
        Creates numericalized input from raw sample
        :param sample: raw sample dictionary
        :param database: database with passages, compatible with sample's indices
        :param tokenizer: model's tokenizer
        :param context_size: size of top-k context the model receives at the input
        :param include_golden_passages: whether to always include golden passage during training
        :param max_input_length: maximum length of each "question|title|passage" input sequence (| marks concatenation)
        :param B:
        :param is_training: whether sample is created for training mode
        :return: numericalized sample(s), note that there can be more, as there can be more answers (or one multi-span answer in case of NQ, treated as more answers)
        """
        title_space_replacement = "_"

        if randomize_context_lengths:
            assert is_training
            if extra_context_size is None:
                context_size = get_random_context_size(2, context_size)
            else:
                new_context_size = get_random_context_size(2, context_size + extra_context_size)
                extra_context_size = new_context_size // 2
                context_size = new_context_size - extra_context_size

                assert extra_context_size >= 0 and context_size >= 0

        if cheat_on_val:
            assert not is_training
            is_training = True

        if is_training and sample["label"] == "NOT ENOUGH INFO":
            jiang_data = None

        if max_input_length is None:
            max_input_length = tokenizer.model_max_length

        if "predicted_article_indices" in sample:
            topk_titles_list = sample['predicted_article_indices']
            topk_titles_list = [unicode_normalize(t) for t in topk_titles_list]
        else:
            # list of top-k predicted indices
            pred_indices = sample["predicted_indices"]

            topk_title_dict = database.get_all(table="documents", columns=["id", "document_title"],
                                               column_name="id", column_value=list(pred_indices[:context_size * B]),
                                               fetch_all=True)
            topk_title_dict = dict(topk_title_dict)
            topk_titles_list = deduplicate_list(topk_title_dict.values())

        # remove disambiguation pages
        topk_titles_list = [title for title in topk_titles_list if not "-LRB-disambiguation-RRB-" in title]

        # taken from
        # https://github.com/dominiksinsaarland/domlin_fever/blob/master/src_legacy/retrieval/filter_lists.py
        def uninformative(title):
            return title.lower().startswith('list_of_') \
                or title.lower().startswith("lists_of_") \
                or title.lower().startswith('index_of_.') \
                or title.lower().startswith('outline_of_')

        # remove lists
        topk_titles_list = [title for title in topk_titles_list if not uninformative(title)]

        ret_docs = database.get_all(table="documents", columns=["id", "document_title", "lines", "lines_i"],
                                    column_name="document_title", column_value=topk_titles_list,
                                    fetch_all=True)

        article_dict_unsorted = defaultdict(lambda: [])
        for _id, doc_title, lines, lines_i in ret_docs:
            article_dict_unsorted[doc_title].append((_id, lines, lines_i))

        article_dict = dict()
        for key, value in dict(article_dict_unsorted).items():
            article_dict[key] = list(sorted(article_dict_unsorted[key], key=lambda x: int(x[-1].split("|")[0])))

        # Throw away too long articles
        for key, value in dict(article_dict).items():
            article_len = sum([len(paragraph[1].split()) for paragraph in value])
            if article_len > 1_500:
                del article_dict[key]

        # firstly fill prepare inputs with true retrieval sample
        top_k_titles = []
        top_k_titles_raw = []

        top_k_passages_tokens = []
        top_k_passages_raw = []
        top_k_sent_ranges = []
        input_passage_ids = []

        gt_evidences_truncated = False
        _iterable = topk_titles_list if "predicted_article_indices" in sample else pred_indices
        # take rest of the passages as top-k, if available
        for topk_item in _iterable:
            # *B because we estimate that there will be ~B blocks per article retrieved
            if len(top_k_passages_tokens) >= context_size * B:
                break
            else:
                title = topk_item if "predicted_article_indices" in sample else topk_title_dict[topk_item]
                try:
                    all_blocks = article_dict[title]
                except KeyError as e:
                    # As MediaWiki searches through current wikipedia, some articles might be missing
                    # or too long articles may be removed
                    continue

                # do not add the same article twice
                if title in top_k_titles_raw:
                    continue

                # preprocess title
                processed_title_from_db = (process_evid(title)) \
                    .replace(title_space_replacement, " ")

                tokenized_title = tokenizer.encode(processed_title_from_db, add_special_tokens=False)

                for _id, lines, lines_i in all_blocks:
                    # sometimes, there can be duplicate passages inside text, remove these cases
                    if lines in top_k_passages_raw:
                        continue

                    preprocessed_passage_from_db = process_evid(lines)

                    # tokenize
                    passage = " " + preprocessed_passage_from_db
                    sentences = passage.split("\n")
                    passage = f" {tokenizer.sentence_special_token} ".join(
                        sentences) + f" {tokenizer.sentence_special_token} "

                    # Also truncate here, as it determines estimate of number of articles
                    tokenized_passage = tokenizer.encode(passage, add_special_tokens=False,
                                                         max_length=max_input_length - len(tokenized_title),
                                                         truncation=True)

                    # keep the record
                    top_k_titles_raw.append(title)
                    top_k_titles.append(tokenized_title)
                    top_k_passages_tokens.append(tokenized_passage)
                    top_k_passages_raw.append(passage)
                    top_k_sent_ranges.append(lines_i)
                    input_passage_ids.append(_id)

        # make sure gt_indices are in top-K context
        # get gt_indices - index of golden passages, if available
        evidences_titles_truncated = False
        try:
            if is_training and sample["label"] != "NOT ENOUGH INFO":
                evidence = deduplicate_list([tuple(ex[2:]) for e in sample['evidence'] for ex in e])
                # retrieve evidence from DB
                all_gt_titles = set(unicode_normalize(gt_title) for gt_title, gt_sent in evidence)

                if jiang_data is not None:
                    jiang_supervision = set(item[0] for item in jiang_data[sample['id']])
                    assert all(s in jiang_supervision for s in list(all_gt_titles))
                    all_gt_titles = jiang_supervision
                    max_gt_articles *= 2

                if truncate_gts and len(all_gt_titles) > max_gt_articles:
                    _allgts = list(all_gt_titles)
                    shuffle(_allgts)
                    all_gt_titles = set(_allgts[:max_gt_articles])
                    evidences_titles_truncated = True

                # query = str(tuple(all_gt_titles)) if len(all_gt_titles) > 1 else f"('{list(all_gt_titles)[0]}')"
                gts_to_insert = database.get_all(table="documents",
                                                 columns=["id", "document_title", "lines", "lines_i"],
                                                 column_name="document_title", column_value=list(all_gt_titles),
                                                 fetch_all=True)
                gt_article_dict_unsorted = defaultdict(lambda: [])
                for _id, doc_title, lines, lines_i in gts_to_insert:
                    gt_article_dict_unsorted[doc_title].append((_id, lines, lines_i))

                gt_article_dict = dict()
                for key, value in dict(gt_article_dict_unsorted).items():
                    gt_article_dict[key] = list(sorted(value, key=lambda x: int(x[-1].split("|")[0])))

                gts_to_insert_ids = set(x[0] for x in gts_to_insert)

                # inject missing evidence into inputs
                if include_golden_passages:
                    # delete all gts from input ids if they are at the boundary of context size (and re-add them again)
                    while top_k_titles_raw[context_size - 1] in gt_article_dict.keys() and \
                            top_k_titles_raw[context_size - 1] == top_k_titles_raw[context_size]:
                        title_to_del_from_inps = top_k_titles_raw[context_size]
                        # min because sometimes there can be less than context_size * B inputs retrieved
                        for idx in range(min(context_size * B, len(top_k_titles_raw)) - 1, -1, -1):
                            if top_k_titles_raw[idx] == title_to_del_from_inps:
                                del input_passage_ids[idx]
                                del top_k_titles_raw[idx]
                                del top_k_titles[idx]
                                del top_k_passages_tokens[idx]
                                del top_k_passages_raw[idx]
                                del top_k_sent_ranges[idx]

                    # until not all gts are in input or
                    # until not all inputs are gts already
                    max_trials = 500
                    trials = 0
                    trials_exceeded = False
                    while not all(gt_id in input_passage_ids[:context_size] for gt_id in gts_to_insert_ids) and \
                            not all(_id in gts_to_insert_ids for _id in input_passage_ids[:context_size]) and \
                            trials < max_trials:  # e.g. rarely it can happen that article has too many passages to fit in
                        trials += 1
                        # go over all articles
                        for gt_article_title in gt_article_dict.keys():
                            # replace randomly selected non-gt article
                            # with equal or longer length
                            if gt_article_title not in top_k_titles_raw[:context_size]:
                                # replace longer (multi-block) articles more often,
                                # as they are included multiple times in top_k_titles_raw

                                gt_article_data = gt_article_dict[gt_article_title]
                                passages_to_insert = [x[0] for x in gt_article_data]

                                max_range = len(input_passage_ids[:context_size]) - len(passages_to_insert)
                                if max_range < 0:
                                    # if the article is too long to insert, just skip it and try the next one
                                    continue

                                r = random.randint(0, max_range)
                                max_trials_local = 500
                                local_trials = 0
                                while any(input_passage_ids[r + i] in gts_to_insert_ids for i in
                                          range(len(passages_to_insert))) \
                                        and local_trials < max_trials_local:  # do not replace gt with gt!
                                    local_trials += 1
                                    r = random.randint(0, max_range)

                                if not local_trials < max_trials_local and not trials_exceeded:
                                    logger.warning("Trials exceeded, could not fit gt article into input. "
                                                   "Increase context size, if this happens too often")
                                    trials_exceeded = True  # Avoid spamming this, if happens multiple times per sample
                                    continue
                                processed_title_from_db = process_evid(gt_article_title) \
                                    .replace(title_space_replacement, " ")

                                tokenized_title = tokenizer.encode(processed_title_from_db, add_special_tokens=False)

                                for idx, (_id, lines, lines_i) in enumerate(gt_article_data):
                                    preprocessed_passage_from_db = process_evid(lines)
                                    passage = " " + preprocessed_passage_from_db
                                    sentences = passage.split("\n")
                                    passage = f" {tokenizer.sentence_special_token} ".join(
                                        sentences) + f" {tokenizer.sentence_special_token} "
                                    tokenized_passage = tokenizer.encode(passage, add_special_tokens=False)

                                    input_passage_ids[r + idx] = _id
                                    top_k_titles_raw[r + idx] = gt_article_title
                                    top_k_titles[r + idx] = tokenized_title
                                    top_k_passages_tokens[r + idx] = tokenized_passage
                                    top_k_passages_raw[r + idx] = passage
                                    top_k_sent_ranges[r + idx] = lines_i
                    if trials == max_trials:
                        trials_exceeded = True
                    gt_evidences_truncated = (all(
                        _id in gts_to_insert_ids for _id in input_passage_ids[:context_size]) and
                                              len(input_passage_ids[:context_size]) < len(gts_to_insert_ids)) or \
                                             trials_exceeded or evidences_titles_truncated
        except Exception as e:
            logger.error(
                f"Error while processing sample for ctx size {context_size} and extra ctx size {extra_context_size}")
            raise e
        top_k_titles_raw = top_k_titles_raw[:context_size]
        top_k_titles = top_k_titles[:context_size]
        top_k_passages_tokens = top_k_passages_tokens[:context_size]
        top_k_passages_raw = top_k_passages_raw[:context_size]
        top_k_sent_ranges = top_k_sent_ranges[:context_size]
        input_passage_ids = input_passage_ids[:context_size]

        jiang_evidences = jiang_data[sample['id']] if jiang_data is not None else None

        if cheat_on_val:
            assert is_training
            is_training = False

        example = FEVERLRMVerifierDataset.prepare_example(gt_evidences_truncated, input_passage_ids,
                                                          is_training,
                                                          max_input_length, sample, tokenizer,
                                                          top_k_passages_tokens, top_k_sent_ranges, top_k_titles,
                                                          top_k_titles_raw, jiang_evidences)
        if hyperlinks_per_sentence is not None:
            current_evidence = [(title, int(sent_range)) for sent_ranges, title in
                                zip(example['metadata']['sent_ranges'], top_k_titles_raw) for sent_range in
                                sent_ranges.split("|")]
            documents = []
            for kth_predicted_evidence in current_evidence:
                documents += list(hyperlinks_per_sentence.get(kth_predicted_evidence, []))
            documents = deduplicate_list(documents)
            documents = [d for d in documents if not d in top_k_titles_raw]

            _sample = copy.deepcopy(sample)
            _sample["predicted_article_indices"] = documents

            hyperlink_example, topk_list_2 = FEVERLRMVerifierDataset.process_sample_perart(_sample,
                                                                                           database,
                                                                                           tokenizer,
                                                                                           extra_context_size,
                                                                                           include_golden_passages,
                                                                                           max_input_length,
                                                                                           False,  # is_training
                                                                                           False,  # cheat_on_val
                                                                                           False,
                                                                                           # randomize_context_lengths
                                                                                           B,
                                                                                           max_gt_articles,
                                                                                           jiang_data,
                                                                                           truncate_gts,
                                                                                           hyperlinks_per_sentence=None)
            example, top_k_titles_raw = FEVERLRMVerifierDataset.merge_processed_examples(
                (example, hyperlink_example),
                (top_k_titles_raw, topk_list_2))

        return example, top_k_titles_raw

    @staticmethod
    def merge_processed_examples(samples, topk_lists):
        is_test_set = 'label' not in samples[0]

        merged_sample = {
            "sources": samples[0]['sources'] + samples[1]['sources'],
            "source_type_ids": samples[0]['source_type_ids'] + samples[1]['source_type_ids'],
            "metadata": {
                "id": samples[0]['metadata']['id'],
                "claim": samples[0]['metadata']['claim'],
                "evidence": samples[0]['metadata']['evidence'],
                "sent_ranges": samples[0]['metadata']['sent_ranges'] + samples[1]['metadata']['sent_ranges'],
                "relevant_sentence_labels": samples[0]['metadata']['relevant_sentence_labels'],
                # there are no relevant labels annotated within "relevant_sentence_labels
            }
        }
        if not is_test_set:
            merged_sample["label"] = samples[0]['label']

        if 'titles' in samples[0]['metadata']:
            merged_sample['metadata']['titles'] = samples[0]['metadata']['titles'] + samples[1]['metadata']['titles']
        if 'irrelevant_sentence_labels' in samples[0]['metadata']:
            merged_sample['metadata']['irrelevant_sentence_labels'] = samples[0]['metadata'][
                'irrelevant_sentence_labels']
        return merged_sample, topk_lists[0] + topk_lists[1]

    @staticmethod
    def prepare_example(evidences_truncated, input_passage_ids, is_training, max_input_length, sample,
                        tokenizer, top_k_passages_tokens, top_k_sent_ranges, top_k_titles,
                        top_k_titles_raw, jiang_evidences):
        # if len(top_k_passages_tokens) != context_size:
        #     logger.warning(
        #         "Not enough passages! This might happen rarely when retrieval won't return too much passages. "
        #         "Be sure it won't happen 'too often'")
        # assert len(top_k_passages_tokens) == context_size, \
        #     f"Passages: {len(top_k_passages_tokens)}, Context size: {context_size}"
        assert len(set(input_passage_ids)) == len(input_passage_ids)  # check there are no duplicates
        claim_r = sample["claim"]
        claim_tokens = tokenizer.encode(claim_r, add_special_tokens=False)
        input_sequences, input_type_ids, sentence_counts, truncation_flags = FEVERLRMVerifierDataset.assemble_input_sequences(
            claim=claim_tokens,
            passages=top_k_passages_tokens,
            titles=top_k_titles,
            tokenizer=tokenizer,
            max_input_length=max_input_length)
        # Compare sentence counts to expected sentence counts.
        # Some sentence could have been truncated out. Remove these from the annotation
        new_topk_sent_ranges = []
        for sentence_count, truncation_status, sentence_id_annotation in \
                zip(sentence_counts, truncation_flags, top_k_sent_ranges):
            sent_ranges = [int(x) for x in sentence_id_annotation.split("|")]
            sentence_count_in_annotation = len(sent_ranges)
            if truncation_status and sentence_count < sentence_count_in_annotation:
                sent_ranges = sent_ranges[:sentence_count]
                sentence_count_in_annotation = len(sent_ranges)
            assert sentence_count == sentence_count_in_annotation
            new_topk_sent_ranges.append("|".join([str(x) for x in sent_ranges]))
        top_k_sent_ranges = new_topk_sent_ranges
        # compute which sentences are relevant
        # [(passage_index, sentence_index), ...]
        relevant_sentence_labels = []
        irrelevant_sentence_labels = []

        is_test_set = 'label' not in sample
        if not is_test_set and sample["label"] != "NOT ENOUGH INFO":
            evidence = deduplicate_list([tuple(ex[2:]) for e in sample['evidence'] for ex in e])
            if jiang_evidences is not None:
                positive_evidence = [
                    (article_id, sentence_id) for article_id, sentence_id, relevancy in jiang_evidences if relevancy]
                assert set(evidence) == set(positive_evidence)
                negative_evidence = [[article_id, sentence_id] for article_id, sentence_id, relevancy in jiang_evidences
                                     if not relevancy]
            for passage_index, (title, sentr) in enumerate(zip(top_k_titles_raw, top_k_sent_ranges)):
                for evidence_title_id, evidence_sent_id in evidence:
                    if title == evidence_title_id:
                        sentence_ids = [int(i) for i in sentr.split("|")]
                        if evidence_sent_id in sentence_ids:
                            relevant_sentence_labels.append((passage_index, sentence_ids.index(evidence_sent_id)))
                if jiang_evidences is not None:
                    for evidence_title_id, evidence_sent_id in negative_evidence:
                        if title == evidence_title_id:
                            sentence_ids = [int(i) for i in sentr.split("|")]
                            if evidence_sent_id in sentence_ids:
                                irrelevant_sentence_labels.append((passage_index, sentence_ids.index(evidence_sent_id)))

            if is_training and not evidences_truncated and not any(truncation_flags):
                assert len(relevant_sentence_labels) >= len(evidence)
                if jiang_evidences is not None:
                    assert len(irrelevant_sentence_labels) >= len(evidence)
        if is_test_set:
            example = {
                "sources": input_sequences,
                "source_type_ids": input_type_ids,
                "metadata": {
                    "id": sample["id"],
                    "claim": sample["claim"],
                    "evidence": sample.get('evidence', None),
                    "sent_ranges": top_k_sent_ranges,
                    "relevant_sentence_labels": relevant_sentence_labels
                }
            }
        else:
            example = {
                "sources": input_sequences,
                "source_type_ids": input_type_ids,
                "label": sample['label'],
                "metadata": {
                    "evidences_truncated": evidences_truncated,  # STATISTIC
                    "id": sample["id"],
                    "claim": sample["claim"],
                    "evidence": sample.get('evidence', None),
                    "sent_ranges": top_k_sent_ranges,
                    "relevant_sentence_labels": relevant_sentence_labels
                }
            }
        if not is_training:
            example['metadata']["titles"] = top_k_titles_raw
        if jiang_evidences is not None:
            example['metadata']['irrelevant_sentence_labels'] = irrelevant_sentence_labels
            example['metadata']['jiang_evidences'] = jiang_evidences
        return example
