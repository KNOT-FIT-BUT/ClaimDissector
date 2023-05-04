# Written by Martin Fajcik <martin.fajcik@vut.cz>
#
import json
import logging
import multiprocessing
import os
import pickle
import time
from functools import partial
from math import ceil
from multiprocessing import Pool
from random import shuffle
from typing import List, AnyStr, Optional, Union

import torch.distributed as dist
from jsonlines import jsonlines
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, RobertaTokenizerFast, RobertaTokenizer

from ..tokenizer.init_tokenizer import init_tokenizer
from ....common.utility import count_lines, unicode_normalize, mkdir

logger = logging.getLogger(__name__)


def preprocessing_method_parallel_wrapper(preprocessing_method, example):
    global kwargs
    return preprocessing_method(example, **kwargs)


def init(_kwargs):
    global kwargs
    _kwargs['tokenizer'] = init_tokenizer(_kwargs['tokenizer'])
    import spacy

    # Disable tokenization errors
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    kwargs = _kwargs


def process_evid(s):
    return unicode_normalize(s)


class RealFCLRMVerifierDataset(IterableDataset):
    def __init__(self, data_file: AnyStr, tokenizer: PreTrainedTokenizer,
                 transformer, block_size, max_len=None,
                 is_training=True, shuffle=False,
                 cache_dir='.data/lrm_verifier', distributed_settings=None, max_context_size=15):
        self.cache_dir = cache_dir
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.max_len = max_len
        self.block_size = block_size
        self.is_training = is_training
        self.shuffle = shuffle
        self.distributed_settings = distributed_settings
        self.max_context_size = max_context_size

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
                               f"_MCS{self.max_context_size}" \
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
                'transformers_cache': ".Transformers_cache"
            },
                "max_input_length": self.max_len,
                "max_context_size": self.max_context_size,
                "is_training": self.is_training,
            }
            log_first = True
            sentence_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sentence_special_token)

            with Pool(processes=num_processes, initializer=init, initargs=[kwargs]) as pool:
                for idx, article_sample in pbar:
                    for idx2, section_name in enumerate(article_sample['labels'].keys()):
                        sample = {
                            "claim": article_sample['claim'],
                            'ncid': article_sample['ncid'],
                            'labels': article_sample['labels'][section_name],
                            'text': article_sample['text'][section_name],
                            'section_name': section_name,
                            'title': article_sample['title']
                        }
                        buffer.append(sample)
                        sample["id"] = f"{'train' if self.is_training else 'd'}_{str(idx)}_{str(idx2)}"
                        if len(buffer) == num_processes:
                            result = pool.map(
                                partial(preprocessing_method_parallel_wrapper, RealFCLRMVerifierDataset.process_sample),
                                buffer)

                            # kwargs = {
                            #     "tokenizer": self.tokenizer,
                            #     "max_input_length": self.max_len,
                            # }
                            # result = [RealFCLRMVerifierDataset.process_sample(e, **kwargs) for e in buffer]

                            total_sentences += sum(
                                [s.count(sentence_special_token) for x in result for s in x['sources']])
                            total_samples += len(buffer)

                            self.write_processed(log_first, result, wf)
                            log_first = False
                            buffer = []
                if len(buffer) > 0:
                    result = pool.map(
                        partial(preprocessing_method_parallel_wrapper, RealFCLRMVerifierDataset.process_sample),
                        buffer)
                    total_samples += len(buffer)
                    self.write_processed(log_first, result, wf)
                    buffer = []

            logger.info(f"Average # of sentences at input {total_sentences / total_samples:.2f}")

    def write_processed(self, log_first, result, wf):
        for example in result:
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
                        # if there is sentence token on -3, there should not be another sentence token on -2 (this can happen bcs of truncation)
                        seq = seq[:max_input_length - 2]
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
                    # on -1 there will be SEP token
                    # on -2 there should be last sentence token
                    if not seq[max_input_length - 3] == sentence_special_token:
                        seq = seq[:max_input_length - 2] + [sentence_special_token]
                    else:
                        # if there is sentence token on -3, there should not be another sentence token on -2 (this can happen bcs of truncation)
                        seq = seq[:max_input_length - 2]
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
    def group_sentences_to_blocks(sentences: List[AnyStr],
                                  tokenizer: PreTrainedTokenizer,
                                  block_size: Union[int, None] = None):
        blocks = []
        accumulated_sentences = []
        RESERVED = 7  # reserve for tokenization artifacts due context dependencies
        accumulated_length = RESERVED
        extra_tokens_len = 3 if tokenizer.name_or_path == 'textattack/roberta-base-MNLI' else 1

        for s in sentences:
            sentence_len = len(tokenizer.tokenize(s)) + extra_tokens_len
            # if this line is super big (larger than block size)
            if sentence_len > block_size:
                # write what was in the buffer, if anything, clear buffer and metadata
                if accumulated_sentences:
                    blocks.append(accumulated_sentences)
                    accumulated_sentences = []
                    accumulated_length = RESERVED
                # write out the big line
                blocks.append([s])
                continue

            # if this line is not longer than block size
            accumulated_length += sentence_len

            # check if accumulating this line would cross block size
            if accumulated_length > block_size:
                # if it would, write out the accumulated lines so far except for this one
                blocks.append(accumulated_sentences)
                # and initialize accumulated set with this line
                accumulated_sentences = [s]
                accumulated_length = sentence_len + RESERVED
            else:
                # otherwise add line to accumulated lines
                accumulated_sentences.append(s)

        # Write out last sentences for document
        if len(accumulated_sentences) > 0:
            blocks.append(accumulated_sentences)

        # sanity check, make sure all lines were used
        assert sum(len(b) for b in blocks) == len(sentences)

        return blocks

    @staticmethod
    def process_sample(sample: dict,
                       tokenizer: PreTrainedTokenizer,
                       is_training: bool,
                       max_input_length: Union[int, None] = None,
                       max_context_size: Union[int, None] = None):
        if max_input_length is None:
            max_input_length = tokenizer.model_max_length

        # firstly fill prepare inputs with true retrieval sample
        top_k_titles = []

        top_k_passages_tokens = []
        top_k_passages_raw = []

        tokenized_claim = tokenizer.encode(unicode_normalize(sample['claim']), add_special_tokens=False)
        title = unicode_normalize(sample['title'] + ': ' + sample['text']['title'])
        tokenized_title = tokenizer.encode(title,
                                           add_special_tokens=False)
        non_empty_sentences = [unicode_normalize(s) for s in sample['text']['sentences'] if s]
        non_empty_sentence_labels = [l for l, s in zip(sample['labels']['sentence_labels'], sample['text']['sentences'])
                                     if s]

        blocks = RealFCLRMVerifierDataset.group_sentences_to_blocks(non_empty_sentences, tokenizer,
                                                                    block_size=max_input_length - len(
                                                                        tokenized_claim) - len(tokenized_title) - 3)
        if len(blocks) > max_context_size:
            if is_training:
                blocks = blocks[:max_context_size]
                total_sentences_left = sum(len(b) for b in blocks)
                # non_empty_sentences = non_empty_sentences[:total_sentences_left]
                non_empty_sentence_labels = non_empty_sentence_labels[:total_sentences_left]
            else:
                # This would be cheating
                raise ValueError("Too many blocks for inference")
        # take rest of the passages as top-k, if available
        for idx, block in enumerate(blocks):
            # tokenize
            sentences = block
            passage = f" {tokenizer.sentence_special_token} ".join(
                sentences) + f" {tokenizer.sentence_special_token}"

            tokenized_passage = tokenizer.encode(passage, add_special_tokens=False)

            # keep the record
            top_k_titles.append(tokenized_title)
            top_k_passages_tokens.append(tokenized_passage)
            top_k_passages_raw.append(passage)

        example = RealFCLRMVerifierDataset.prepare_example(tokenized_claim,
                                                           non_empty_sentence_labels,
                                                           max_input_length, sample, tokenizer,
                                                           top_k_passages_tokens, top_k_titles, title)
        return example

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
    def prepare_example(tokenized_claim,
                        non_empty_sentence_labels,
                        max_input_length, sample, tokenizer,
                        top_k_passages_tokens, top_k_titles, title):

        input_sequences, input_type_ids, sentence_counts, truncation_flags = RealFCLRMVerifierDataset.assemble_input_sequences(
            claim=tokenized_claim,
            passages=top_k_passages_tokens,
            titles=top_k_titles,
            tokenizer=tokenizer,
            max_input_length=max_input_length)
        example = {
            "sources": input_sequences,
            "source_type_ids": input_type_ids,
            "label": sample['labels']['section_label'],
            "metadata": {
                "id": sample["id"],
                "ncid": sample["ncid"],
                "section_name": sample["section_name"],
                "claim": sample["claim"],
                "title": title,
                "sentence_labels": non_empty_sentence_labels,
            }
        }
        return example
