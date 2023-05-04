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
from math import ceil, floor
from multiprocessing import Pool
from random import shuffle
from typing import List, AnyStr, Optional, Union

import torch
import torch.distributed as dist
from jsonlines import jsonlines
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, RobertaTokenizerFast, RobertaTokenizer

from .fever_verifier_dataset import FEVERLRMVerifierDataset
from ..tokenizer.init_tokenizer import init_tokenizer
from ....common.db import PassageDB
from ....common.eval_utils import f1_score
from ....common.utility import count_lines, unicode_normalize, deduplicate_list, mkdir

logger = logging.getLogger(__name__)


def has_sent_hit(example, topk_titles_raw):
    if not example['metadata']['evidence'] or len(example['metadata']['orig_sample']['supporting_facts']) == 0:
        return False
    pred_id_sent_list = [[int(x) for x in y.split("|")] for y in example["metadata"]['sent_ranges']]
    # hover contains just 1 annotated group, as far as I understand
    sent_hits = []
    # for every annotated item in single annotation
    for doc_title, sent_id in example['metadata']['evidence']:
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


class HoverLRMVerifierDataset(IterableDataset):
    def __init__(self, data_file: AnyStr, tokenizer: PreTrainedTokenizer,
                 transformer, database, context_length, block_size, max_len=None,
                 is_training=True, include_golden_passages=True, shuffle=False,
                 eval_interpretability=False, cheat_on_val=False, khattab_like=False,
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
        self.include_golden_passages = include_golden_passages
        self.distributed_settings = distributed_settings
        self.khattab_like = khattab_like

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
        without_psg_suffix = f"_withoutpassages" if not self.include_golden_passages else ""
        maxlen = f'_L{self.max_len}' if self.max_len is not None else ''
        mkdir(self.cache_dir)
        preprocessed_f_noext = os.path.join(self.cache_dir, os.path.basename(
            self.data_file['official'])) + f"_verifier_preprocessed_for" \
                                           f"_{transformer}" \
                                           f"_C{self.context_length}" \
                                           f"_B{self.block_size}" \
                                           f"{without_psg_suffix}" \
                                           f"{'_cheat_on_val' if self.cheat_on_val else ''}" \
                                           f"{'_interp' if self.eval_interpretability else ''}" \
                                           f"{'_khattablike' if self.khattab_like else ''}" \
                                           f"{maxlen}"
        preprocessed_f = preprocessed_f_noext + ".jsonl"
        return preprocessed_f

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
        num_lines = count_lines(self.data_file["retrieval_inputs"])
        number_of_short_contexts = 0
        accumulated_context_length = 0
        total_samples = 0

        hits_at_K = 0
        mh_hits_at_K = [0, 0, 0]
        total_multihop_samples = [0, 0, 0]
        total_enoughinfo_samples = 0

        supported_hits_at_K = 0
        supported_mh_hits_at_K = [0, 0, 0]
        supported_total_enoughinfo_samples = 0
        supported_total_multihop_samples = [0, 0, 0]

        CLASS_STATISTICS = {
            "SUPPORTED": 0,
            "NOT_SUPPORTED": 0
        }

        total_sentences = 0
        total_articles = 0
        num_processes = multiprocessing.cpu_count()

        with open(self.data_file["official"], "r") as rf:
            official_dataset = json.load(rf)
        with open(self.data_file["retrieval"], "r") as rf:
            retrieval_results = json.load(rf)
        assert len(official_dataset) == len(retrieval_results) == num_lines
        with jsonlines.open(self.data_file["retrieval_inputs"], "r") as reader_retrieval_inputs, \
                jsonlines.open(self.preprocessed_f, "w") as wf:
            pbar = tqdm(enumerate(zip(official_dataset, reader_retrieval_inputs)), total=num_lines)
            buffer = []
            mh_status_buffer = []
            supportclass_status_buffer = []
            kwargs = {"database": self.database.path,
                      "tokenizer": {
                          'verifier_tokenizer_type': self.tokenizer.name_or_path,
                          'transformers_cache': ".Transformers_cache"
                      },
                      "max_input_length": self.max_len,
                      "context_size": self.context_length,
                      "include_golden_passages": self.include_golden_passages,
                      "is_training": self.is_training,
                      "cheat_on_val": self.cheat_on_val,
                      "khattab_like": self.khattab_like
                      }
            log_first = True
            sentence_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sentence_special_token)

            with Pool(processes=num_processes, initializer=init, initargs=[kwargs]) as pool:
                for idx, (official_sample, retrieval_input_sample) in pbar:
                    is_test_set = not 'label' in official_sample
                    # Retrieval inputs differ slightly (e.g. double spaces), for unknown reason
                    assert f1_score(official_sample["claim"], retrieval_input_sample["question"]) > 0.95
                    if not is_test_set:
                        if self.eval_interpretability and official_sample['uid'] not in self.data_subset:
                            continue
                        CLASS_STATISTICS[official_sample["label"]] += 1
                        num_of_hops = official_sample['num_hops']
                        if len(official_sample['supporting_facts']) > 0:
                            total_enoughinfo_samples += 1
                            if official_sample['label'] == "SUPPORTED":
                                supported_total_enoughinfo_samples += 1
                        if num_of_hops > 0:
                            total_multihop_samples[num_of_hops - 2] += 1
                            if official_sample['label'] == "SUPPORTED":
                                supported_total_multihop_samples[num_of_hops - 2] += 1
                            mh_status_buffer.append(num_of_hops)
                        supportclass_status_buffer.append(official_sample['label'] == "SUPPORTED")
                    buffer.append({
                        "sample": official_sample,
                        "retrieved": retrieval_results[str(retrieval_input_sample["qid"])]
                    })

                    if len(buffer) == num_processes:
                        result = pool.map(
                            partial(preprocessing_method_parallel_wrapper,
                                    HoverLRMVerifierDataset.process_sample_perart),
                            buffer)
                        # DBG
                        # _kwargs = {
                        #     "database": self.database,
                        #     "tokenizer": self.tokenizer,
                        #     "max_input_length": self.max_len,
                        #     "context_size": self.context_length,
                        #     "include_golden_passages": self.include_golden_passages,
                        #     "is_training": self.is_training,
                        # }
                        # kwargs.update(_kwargs)
                        # result = [HoverLRMVerifierDataset.process_sample_perart(e, **kwargs) for e in buffer]

                        total_sentences += sum(
                            [s.count(sentence_special_token) for x in result for s in x[0]['sources']])
                        total_samples += len(buffer)
                        accumulated_context_length, \
                            hits, \
                            number_of_short_contexts, \
                            total_articles = self.write_processed(
                            accumulated_context_length, log_first, number_of_short_contexts, pbar,
                            result, total_articles, wf)
                        if not is_test_set:
                            hits_at_K += sum(hits)
                            supported_hits_at_K += sum(
                                (h for h, sup_status in zip(hits, supportclass_status_buffer) if sup_status))
                            assert len(hits) == len(mh_status_buffer) == len(supportclass_status_buffer)
                            for h, hops, sup_status in zip(hits, mh_status_buffer, supportclass_status_buffer):
                                mh_hits_at_K[hops - 2] += h
                                if sup_status:
                                    supported_mh_hits_at_K[hops - 2] += h

                            pbar.set_description(
                                f"Supported Sentence-Recall@{self.context_length}: {supported_hits_at_K * 100 / supported_total_enoughinfo_samples:.2f}, "
                                f"Supported MH_Recall@{self.context_length}: {[f'{(mh_hits_at_K_hop * 100 / total_multihop_samples_hop) if total_multihop_samples_hop > 0 else -1:.2f}' for mh_hits_at_K_hop, total_multihop_samples_hop in zip(supported_mh_hits_at_K, supported_total_multihop_samples)]}")
                        log_first = False
                        buffer = []
                        mh_status_buffer = []
                        supportclass_status_buffer = []
                if len(buffer) > 0:
                    result = pool.map(
                        partial(preprocessing_method_parallel_wrapper, HoverLRMVerifierDataset.process_sample_perart),
                        buffer)
                    # kwargs = {  # "sent_mapping": self.sent_mapping,
                    #     "database": self.database,
                    #     "tokenizer": self.tokenizer,
                    #     "max_input_length": self.max_len,
                    #     "context_size": self.context_length,
                    #     "include_golden_passages": self.include_golden_passages,
                    #     "is_training": self.is_training,
                    # }
                    # result = [HoverLRMVerifierDataset.process_sample_perart(e, **kwargs) for e in buffer]
                    total_sentences += sum(
                        [s.count(sentence_special_token) for x in result for s in x[0]['sources']])
                    total_samples += len(buffer)
                    accumulated_context_length, \
                        hits, \
                        number_of_short_contexts, \
                        total_articles = self.write_processed(
                        accumulated_context_length, log_first, number_of_short_contexts, pbar,
                        result, total_articles, wf)
                    if not is_test_set:
                        hits_at_K += sum(hits)
                        supported_hits_at_K += sum(
                            (h for h, sup_status in zip(hits, supportclass_status_buffer) if sup_status))
                        assert len(hits) == len(mh_status_buffer) == len(supportclass_status_buffer)
                        for h, hops, sup_status in zip(hits, mh_status_buffer, supportclass_status_buffer):
                            mh_hits_at_K[hops - 2] += h
                            if sup_status:
                                supported_mh_hits_at_K[hops - 2] += h

                        pbar.set_description(
                            f"Supported Sentence-Recall@{self.context_length}: {supported_hits_at_K * 100 / supported_total_enoughinfo_samples:.2f}, "
                            f"Supported MH_Recall@{self.context_length}: {[f'{(mh_hits_at_K_hop * 100 / total_multihop_samples_hop) if total_multihop_samples_hop > 0 else -1:.2f}' for mh_hits_at_K_hop, total_multihop_samples_hop in zip(supported_mh_hits_at_K, supported_total_multihop_samples)]}")
                    log_first = False
                    buffer = []
                    mh_status_buffer = []
                    supportclass_status_buffer = []

        if not is_test_set:
            logger.info("Class statistics:\n")
            logger.info(json.dumps(CLASS_STATISTICS, indent=4))
            logger.info(
                f"Overall Sentence hit @ {self.context_length}: {hits_at_K * 100 / total_enoughinfo_samples:.2f}"
                f" ({hits_at_K}/{total_enoughinfo_samples})")
            logger.info(
                f"Overall MH Sentence hit @ {self.context_length}: {[f'{(mh_hits_at_K_hop * 100 / total_multihop_samples_hop) if total_multihop_samples_hop > 0 else -1:.2f}' for mh_hits_at_K_hop, total_multihop_samples_hop in zip(mh_hits_at_K, total_multihop_samples)]}"
                f" ({[f'{mh_hits_at_K_hop}/{total_multihop_samples_hop}' for mh_hits_at_K_hop, total_multihop_samples_hop in zip(mh_hits_at_K, total_multihop_samples)]})")

            logger.info(
                f"SUPPORTED Sentence hit @ {self.context_length}: {supported_hits_at_K * 100 / supported_total_enoughinfo_samples:.2f}"
                f" ({supported_hits_at_K}/{supported_total_enoughinfo_samples})")
            logger.info(
                f"SUPPORTED MH Sentence hit @ {self.context_length}: {[f'{(mh_hits_at_K_hop * 100 / total_multihop_samples_hop) if total_multihop_samples_hop > 0 else -1:.2f}' for mh_hits_at_K_hop, total_multihop_samples_hop in zip(supported_mh_hits_at_K, supported_total_multihop_samples)]}"
                f" ({[f'{mh_hits_at_K_hop}/{total_multihop_samples_hop}' for mh_hits_at_K_hop, total_multihop_samples_hop in zip(supported_mh_hits_at_K, supported_total_multihop_samples)]})")

            logger.info(f"Average # of sentences at input {total_sentences / total_samples:.2f}")
            logger.info(f"Average # of articles at input {total_articles / total_samples:.2f}")
            logger.info(f"Average context_length: {accumulated_context_length / total_samples:.2f}")
            logger.info(f"Number of contexts shorter than context_length {number_of_short_contexts}")

            logger.info(f"Total samples with evidence from total {total_enoughinfo_samples}/{total_samples}")
            logger.info(
                f"Total multihop samples: {sum(total_multihop_samples)}/{total_samples} "
                f"({sum(total_multihop_samples) / total_samples * 100 :.2f}%)")
        else:
            logger.info(f"Average # of sentences at input {total_sentences / total_samples:.2f}")
            logger.info(f"Average # of articles at input {total_articles / total_samples:.2f}")
            logger.info(f"Average context_length: {accumulated_context_length / total_samples:.2f}")
            logger.info(f"Number of contexts shorter than context_length {number_of_short_contexts}")

    def write_processed(self, accumulated_context_length, log_first, number_of_short_contexts, pbar,
                        result, total_articles, wf):
        hits = []
        for example, topk_titles_raw in result:
            is_test_set = 'label' not in example
            if not is_test_set:
                total_articles += len(set(topk_titles_raw))
                if len(example['metadata']['orig_sample']['supporting_facts']) > 0:
                    hits.append(has_sent_hit(example, topk_titles_raw))
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
        return accumulated_context_length, hits, number_of_short_contexts, total_articles

    @staticmethod
    def process_sample_perart(sample: dict,
                              database: Union[PassageDB, AnyStr],
                              tokenizer: PreTrainedTokenizer,
                              context_size: int,
                              include_golden_passages=True,
                              max_input_length: Union[int, None] = None,
                              is_training: bool = False,
                              cheat_on_val: bool = False,
                              B: int = 6,
                              khattab_like=False,
                              max_gt_articles=15,
                              truncate_gts=True):
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
        if cheat_on_val:
            assert not is_training
            is_training = True

        if max_input_length is None:
            max_input_length = tokenizer.model_max_length

        official_sample = sample['sample']
        retrieved_for_sample = sample['retrieved'][1]

        # For comparison with Condenser
        condenser_predicted = sample['retrieved'][0]

        if type(retrieved_for_sample[0]) == list:
            # list of top-k predicted indices for every hop
            NUM_HOPS = 4  # assuming 4 hop retrieval for hoVER
            perhop_context_size = floor(context_size / NUM_HOPS)

            """
            Combine perhop_context_size documents from multiple hops, as we will filter some out
            take always perhop_context_size from each document, (padded with 1st document retrieval results if not divisible)
            
            Do this multiple times so we have enough documents
            """
            pred_indices = []
            for i in range(B):
                for retrieved_list in retrieved_for_sample:
                    pred_indices += retrieved_list[perhop_context_size * i:perhop_context_size * (i + 1)]

                # pad the rest from first hop retrieval
                remainder = context_size % NUM_HOPS
                pred_indices += retrieved_for_sample[0][remainder * i:remainder * (i + 1)]
        else:
            pred_indices = retrieved_for_sample
        topk_title_dict = database.get_all(table="documents", columns=["pid", "document_title"],
                                           column_name="pid", column_value=list(pred_indices[:context_size * B]),
                                           fetch_all=True)
        topk_title_dict = dict(topk_title_dict)
        topk_titles_list = deduplicate_list(topk_title_dict.values())

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

        # if there are multiple blocks with the same doc-title, sort them
        article_dict = dict()
        for key, value in dict(article_dict_unsorted).items():
            try:
                article_dict[key] = list(sorted(article_dict_unsorted[key], key=lambda x: int(x[-1].split("|")[0])))
            except ValueError as e:
                logging.error(key)
                logging.error(str(article_dict_unsorted[key]))
                raise e
        # Disabled for HOVER for now
        # Throw away too long articles
        # for key, value in dict(article_dict).items():
        #     article_len = sum([len(paragraph[1].split()) for paragraph in value])
        #     if article_len > 1_500:
        #         del article_dict[key]

        # firstly fill prepare inputs with true retrieval sample
        top_k_titles = []
        top_k_titles_raw = []

        top_k_passages_tokens = []
        top_k_passages_raw = []
        top_k_sent_ranges = []
        input_passage_ids = []

        gt_evidences_truncated = False
        _iterable = pred_indices
        # take rest of the passages as top-k, if available
        for topk_item in pred_indices:
            # *B because we estimate that there will be ~B blocks per article retrieved
            if len(top_k_passages_tokens) >= context_size * B:
                break
            else:
                try:
                    title = topk_title_dict[topk_item]
                except KeyError:
                    logging.warning(f"Item {topk_item} not found in block database!")
                    continue
                try:
                    all_blocks = article_dict[title]
                except KeyError:
                    # The title was removed in filtering
                    if title not in topk_titles_list:
                        continue
                    else:
                        # No other keys should not be missing in HoVER with Baleen!
                        logging.debug(title)
                        logging.debug(str(article_dict))

                # do not add the same article twice
                if title in top_k_titles_raw:
                    continue

                # preprocess title
                processed_title_from_db = unicode_normalize(title)

                tokenized_title = tokenizer.encode(processed_title_from_db, add_special_tokens=False)

                for _id, lines, lines_i in all_blocks:
                    # sometimes, there can be duplicate passages inside text, remove these cases
                    if lines in top_k_passages_raw:
                        continue

                    preprocessed_passage_from_db = unicode_normalize(lines)

                    # tokenize
                    passage = " " + preprocessed_passage_from_db
                    sentences = passage.split("\n")
                    # HoVer sometimes contains empty sentences, just replace them with "empty" to avoid NaNs
                    sentences = [s if s.strip() else "empty" for s in sentences]

                    passage = f" {tokenizer.sentence_special_token} ".join(
                        sentences) + f" {tokenizer.sentence_special_token} "

                    # Also truncate here, as it determines estimate of number of articles
                    tokenized_passage = tokenizer.encode(passage, add_special_tokens=False,
                                                         max_length=max_input_length - len(tokenized_title),
                                                         truncation=True)
                    # DBG
                    # sentence_special_token_id = tokenizer.convert_tokens_to_ids(
                    #     tokenizer.sentence_special_token)
                    # for i in range(len(tokenized_passage) - 1):
                    #     if tuple(tokenized_passage[i:i + 2]) == (sentence_special_token_id, sentence_special_token_id):
                    #         a = 1
                    #         b = 2

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
        negp_sample_titles = None
        if is_training and len(official_sample['supporting_facts']) > 0:
            evidence = official_sample['supporting_facts']

            # retrieve evidence from DB
            all_gt_titles = set(gt_title for gt_title, gt_sent in evidence)

            if khattab_like:
                # Sample negative passages between top-15 and top-K (this is not exactly like Khattab did it)
                negative_passage_candidate_set = [c for r in retrieved_for_sample for c in r[15:]]
                shuffle(negative_passage_candidate_set)
                negp_sample = negative_passage_candidate_set[:len(all_gt_titles) * 2]
                negp_sample_title_dict = database.get_all(table="documents", columns=["pid", "document_title"],
                                                          column_name="pid",
                                                          column_value=negp_sample,
                                                          fetch_all=True)
                negp_sample_titles = list(dict(negp_sample_title_dict).values())
                shuffle(negp_sample_titles)
                negp_sample_titles = negp_sample_titles[:len(all_gt_titles)]
                all_gt_titles = all_gt_titles.union(negp_sample_titles)

            if truncate_gts and len(all_gt_titles) > max_gt_articles:
                _allgts = list(all_gt_titles)
                shuffle(_allgts)
                all_gt_titles = set(_allgts[:max_gt_articles])
                evidences_titles_truncated = True

            gts_to_insert = database.get_all(table="documents", columns=["id", "document_title", "lines", "lines_i"],
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
            if is_training and include_golden_passages:
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
                                logging.warning("Trials exceeded, could not fit gt article into input. "
                                                "Increase context size, if this happens too often")
                                trials_exceeded = True  # Avoid spamming this, if happens multiple times per sample
                                continue
                            processed_title_from_db = unicode_normalize(gt_article_title)

                            tokenized_title = tokenizer.encode(processed_title_from_db, add_special_tokens=False)

                            for idx, (_id, lines, lines_i) in enumerate(gt_article_data):
                                preprocessed_passage_from_db = unicode_normalize(lines)
                                passage = " " + preprocessed_passage_from_db
                                sentences = passage.split("\n")
                                # HoVer sometimes contains empty sentences, just replace them with "empty" to avoid NaNs
                                sentences = [s if s.strip() else "empty" for s in sentences]

                                passage = f" {tokenizer.sentence_special_token} ".join(
                                    sentences) + f" {tokenizer.sentence_special_token} "
                                tokenized_passage = tokenizer.encode(passage, add_special_tokens=False)

                                # DBG
                                # sentence_special_token_id = tokenizer.convert_tokens_to_ids(
                                #     tokenizer.sentence_special_token)
                                # for i in range(len(tokenized_passage) - 1):
                                #     if tuple(tokenized_passage[i:i + 2]) == (sentence_special_token_id, sentence_special_token_id):
                                #         a = 1
                                #         b = 2

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

        top_k_titles_raw = top_k_titles_raw[:context_size]
        top_k_titles = top_k_titles[:context_size]
        top_k_passages_tokens = top_k_passages_tokens[:context_size]
        top_k_passages_raw = top_k_passages_raw[:context_size]
        top_k_sent_ranges = top_k_sent_ranges[:context_size]
        input_passage_ids = input_passage_ids[:context_size]

        if cheat_on_val:
            assert is_training
            is_training = False

        example = HoverLRMVerifierDataset.prepare_example(gt_evidences_truncated, context_size, input_passage_ids,
                                                          is_training, pred_indices,
                                                          max_input_length, sample, tokenizer,
                                                          top_k_passages_tokens, top_k_sent_ranges, top_k_titles,
                                                          top_k_titles_raw,
                                                          irrelevant_titles=negp_sample_titles,
                                                          condenser_predicted=condenser_predicted)
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
    def prepare_example(evidences_truncated, context_size, input_passage_ids, is_training, pred_indices,
                        max_input_length, sample,
                        tokenizer, top_k_passages_tokens, top_k_sent_ranges, top_k_titles,
                        top_k_titles_raw, irrelevant_titles=None, condenser_predicted=None):
        if len(top_k_passages_tokens) != context_size:
            logging.warning(
                f"Not enough passages {len(top_k_passages_tokens)}/{context_size}! "
                f"Originally there was {len(pred_indices)} retrieved passages. "
                f"This might happen rarely when retrieval won't return too much passages. "
                "Be sure it won't happen 'too often'")
        # assert len(top_k_passages_tokens) == context_size, \
        #     f"Passages: {len(top_k_passages_tokens)}, Context size: {context_size}"
        assert len(set(input_passage_ids)) == len(input_passage_ids)  # check there are no duplicates
        claim_r = sample['sample']["claim"]
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

        official_sample = sample['sample']
        is_test_set = 'label' not in official_sample
        if not is_test_set and len(official_sample['supporting_facts']) > 0:
            evidence = official_sample['supporting_facts']
            for passage_index, (title, sentr) in enumerate(zip(top_k_titles_raw, top_k_sent_ranges)):
                for evidence_title_id, evidence_sent_id in evidence:
                    if title == evidence_title_id:
                        sentence_ids = [int(i) for i in sentr.split("|")]
                        if evidence_sent_id in sentence_ids:
                            relevant_sentence_labels.append((passage_index, sentence_ids.index(evidence_sent_id)))

            if is_training and not evidences_truncated and not any(truncation_flags):
                # assert len(relevant_sentence_labels) >= len(evidence)
                if not len(relevant_sentence_labels) >= len(evidence):
                    logger.warning(
                        f"MISSING EVIDENCES AT INPUT!: # of labels is {len(relevant_sentence_labels)}, while # of evidences is {len(evidence)}"
                        f"(happens in HoVer sometimes due to imperfect label-sentence mapping...)")
        if is_test_set:
            example = {
                "sources": input_sequences,
                "source_type_ids": input_type_ids,
                "metadata": {
                    "id": sample['sample']["uid"],
                    "claim": sample['sample']["claim"],
                    "evidence": sample['sample'].get('supporting_facts', None),
                    "sent_ranges": top_k_sent_ranges,
                    "relevant_sentence_labels": relevant_sentence_labels,
                    "orig_sample": sample['sample'],
                }
            }
        else:
            example = {
                "sources": input_sequences,
                "source_type_ids": input_type_ids,
                "label": sample['sample']['label'],
                "metadata": {
                    "id": sample['sample']["uid"],
                    "claim": sample['sample']["claim"],
                    "evidence": sample['sample'].get('supporting_facts', None),
                    "sent_ranges": top_k_sent_ranges,
                    "relevant_sentence_labels": relevant_sentence_labels,
                    "orig_sample": sample['sample'],
                    "condenser_predicted": condenser_predicted,
                }
            }
        example['metadata']["titles"] = top_k_titles_raw
        if irrelevant_titles is not None:
            example['metadata']['irrelevant_titles'] = irrelevant_titles
        return example
