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
from ....common.utility import count_lines, unicode_normalize, deduplicate_list, mkdir

logger = logging.getLogger(__name__)


def preprocessing_method_parallel_wrapper(preprocessing_method, example):
    global kwargs
    return preprocessing_method(example, **kwargs)


def init(_kwargs):
    global kwargs
    _kwargs['tokenizer'] = init_tokenizer(_kwargs['tokenizer'])
    import spacy
    _kwargs['sentencizer'] = spacy.load("en_core_web_sm", disable=['ner', 'tagger', 'attribute_ruler', 'lemmatizer'])

    # Disable tokenization errors
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    kwargs = _kwargs


def process_evid(s):
    return unicode_normalize(s)


class FaviqLRMVerifierDataset(IterableDataset):
    def __init__(self, data_file: AnyStr, tokenizer: PreTrainedTokenizer,
                 transformer, block_size, max_len=None,
                 is_training=True, shuffle=False, randomize_context_lengths=False,
                 cache_dir='.data/lrm_verifier', distributed_settings=None):
        self.cache_dir = cache_dir
        self.data_file = data_file
        self.tokenizer = tokenizer
        import spacy
        self.spacynlp = spacy.load("en_core_web_sm", disable=['ner', 'tagger', 'attribute_ruler', 'lemmatizer'])

        self.transformer = transformer
        self.max_len = max_len
        self.randomize_context_lengths = randomize_context_lengths
        self.block_size = block_size
        self.is_training = is_training
        self.shuffle = shuffle
        self.distributed_settings = distributed_settings

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
        maxlen = f'_L{self.max_len}' if self.max_len is not None else ''
        mkdir(self.cache_dir)
        preprocessed_f_noext = os.path.join(self.cache_dir, os.path.basename(
            self.data_file)) + f"_verifier_preprocessed_for" \
                               f"_{transformer}" \
                               f"_B{self.block_size}" \
                               f"{maxlen}"
        preprocessed_f = preprocessed_f_noext + ".jsonl"
        return preprocessed_f

    def _preprocess_data(self):
        num_lines = count_lines(self.data_file)
        total_sentences = 0
        total_samples = 0

        num_processes = multiprocessing.cpu_count()
        with jsonlines.open(self.data_file, "r") as fd, jsonlines.open(self.preprocessed_f, "w") as wf:
            pbar = tqdm(enumerate(fd), total=num_lines)
            buffer = []
            kwargs = {"tokenizer": {
                'verifier_tokenizer_type': self.tokenizer.name_or_path,
                'transformers_cache': ".Transformers_cache",
            },

                "randomize_context_lengths": self.randomize_context_lengths,
                "max_input_length": self.max_len,
            }
            log_first = True
            sentence_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sentence_special_token)

            with Pool(processes=num_processes, initializer=init, initargs=[kwargs]) as pool:
                for idx, sample in pbar:
                    buffer.append(sample)
                    sample["id"] = f"{'train' if self.is_training else 'd'}_{str(idx)}"
                    if len(buffer) == num_processes:
                        result = pool.map(
                            partial(preprocessing_method_parallel_wrapper, FaviqLRMVerifierDataset.process_sample),
                            buffer)

                        # kwargs = {
                        #     "tokenizer": self.tokenizer,
                        #     "sentencizer": self.spacynlp,
                        #     "max_input_length": self.max_len,
                        #     "is_training": self.is_training,
                        # }
                        # result = [FaviqLRMVerifierDataset.process_sample(e, **kwargs) for e in buffer]

                        total_sentences += sum(
                            [s.count(sentence_special_token) for x in result for s in x[0]['sources']])
                        total_samples += len(buffer)

                        self.write_processed(log_first, result, wf)
                        log_first = False
                        buffer = []
                if len(buffer) > 0:
                    result = pool.map(
                        partial(preprocessing_method_parallel_wrapper, FaviqLRMVerifierDataset.process_sample),
                        buffer)
                    total_samples += len(buffer)
                    self.write_processed(log_first, result, wf)
                    buffer = []

            logger.info(f"Average # of sentences at input {total_sentences / total_samples:.2f}")

    def write_processed(self, log_first, result, wf):
        for example, topk_titles_raw in result:
            wf.write(example)
            if log_first:
                logger.info("Example of model input formats:")

                def log_sample(e):
                    logger.info("*" * 10)
                    src_example1 = " ".join(self.tokenizer.convert_ids_to_tokens(e["sources"][-1]))
                    logger.info("inputs 1:")
                    logger.info(src_example1)

                log_sample(example)

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

            for tidx, (title, passage) in enumerate(zip(titles, passages)):
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
    def get_sentences(sentencizer, block):
        output = sentencizer(block)
        sentences = []
        for i, sent in enumerate(output.sents):
            if sent.text.strip() != "":
                sentences.append(sent.text)
        return sentences

    @staticmethod
    def process_sample(sample: dict,
                       tokenizer: PreTrainedTokenizer,
                       sentencizer,
                       randomize_context_lengths=False,
                       max_input_length: Union[int, None] = None):
        if max_input_length is None:
            max_input_length = tokenizer.model_max_length

        if randomize_context_lengths:
            raise NotImplementedError("Random context size is not implemented for FAVIQ dataset")

        # firstly fill prepare inputs with true retrieval sample
        top_k_titles = []
        top_k_titles_raw = []

        top_k_passages_tokens = []
        top_k_passages_raw = []
        input_passage_ids = []
        relevant_passage_labels = []

        removed = 0
        # take rest of the passages as top-k, if available
        for idx, topk_item in enumerate(sample['ctxs']):
            if topk_item['id'] == "N/A":
                removed += 1
                continue
            if topk_item['has_answer']:
                relevant_passage_labels.append(idx - removed)
            title = topk_item['title']
            # preprocess title
            processed_title_from_db = (process_evid(title))

            tokenized_title = tokenizer.encode(processed_title_from_db, add_special_tokens=False)
            retrieved_passage = process_evid(topk_item['text'])

            # tokenize
            sentences = FaviqLRMVerifierDataset.get_sentences(sentencizer, " " + retrieved_passage)
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

        context_size = 20  # FIXED, following Asai in EvidentialityQA
        top_k_titles_raw = top_k_titles_raw[:context_size]
        top_k_titles = top_k_titles[:context_size]
        top_k_passages_tokens = top_k_passages_tokens[:context_size]
        top_k_passages_raw = top_k_passages_raw[:context_size]
        input_passage_ids = input_passage_ids[:context_size]
        relevant_passage_labels = [l for l in relevant_passage_labels if l < context_size]

        example = FaviqLRMVerifierDataset.prepare_example(input_passage_ids,
                                                          relevant_passage_labels,
                                                          max_input_length, sample, tokenizer,
                                                          top_k_passages_tokens, top_k_titles,
                                                          top_k_titles_raw)
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
    def prepare_example(input_passage_ids, relevant_passage_labels, max_input_length, sample,
                        tokenizer, top_k_passages_tokens, top_k_titles, top_k_titles_raw):

        assert len(set(input_passage_ids)) == len(input_passage_ids)  # check there are no duplicates
        claim_r = sample["question"]
        claim_tokens = tokenizer.encode(claim_r, add_special_tokens=False)
        input_sequences, input_type_ids, sentence_counts, truncation_flags = FaviqLRMVerifierDataset.assemble_input_sequences(
            claim=claim_tokens,
            passages=top_k_passages_tokens,
            titles=top_k_titles,
            tokenizer=tokenizer,
            max_input_length=max_input_length)
        # Compare sentence counts to expected sentence counts.
        # Some sentence could have been truncated out. Remove these from the annotation
        top_k_sent_ranges = ["|".join([str(x) for x in range(l)]) for l in sentence_counts]

        example = {
            "sources": input_sequences,
            "source_type_ids": input_type_ids,
            "label": sample['answers'][0],
            "metadata": {
                "id": sample["id"],
                "claim": sample["question"],
                "sent_ranges": top_k_sent_ranges,
                "relevant_passage_labels": relevant_passage_labels,
                "titles": top_k_titles_raw
            }
        }
        return example
