# Written by Martin Fajcik <martin.fajcik@vut.cz>
#
import copy
import csv
import json
import logging
import math
import os
import pickle
import socket
import time
import uuid
from collections import Counter

import torch.distributed as dist
import numpy as np
import torch
import torch.nn.functional as F
import transformers

from math import ceil
from random import randint, shuffle, random, sample
from fever.scorer import fever_score
from jsonlines import jsonlines
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score as f1_score_metric

from .dataset.hover_lrm_verifier_dataset import HoverLRMVerifierDataset
from .dataset.faviq_verifier_dataset import FaviqLRMVerifierDataset
from .dataset.realfc_verifier_dataset import RealFCLRMVerifierDataset
from .dataset.fever_verifier_dataset import FEVERLRMVerifierDataset
from .evaluation.realfc.evalscript import eval_realfc_official

from ..models.transformer_masker import TransformerMasker
from ..models.transformer_classifier import TransformerClassifier

from .tokenizer.init_tokenizer import init_tokenizer
from .evaluation.hover.eval import hover_eval
from ...common.drqa_tokenizers.simple_tokenizer import SimpleTokenizer
from ...common.db import PassageDB
from ...common.distributed_utils import share_list
from ...common.eval_utils import f1_score
from ...common.utility import get_timestamp, get_model, cat_lists, print_eval_stats, mkdir, report_parameters, \
    sum_parameters, count_parameters, deduplicate_list, get_samples_cumsum, flatten

logger = logging.getLogger(__name__)

gt_dicts = {"hover":
                {"SUPPORTED": 0, "NOT_SUPPORTED": 1},
            "fever": {
                "SUPPORTS": 0,
                "REFUTES": 1,
                "NOT ENOUGH INFO": 2
            },
            "faviq": {
                "SUPPORTS": 0,
                "REFUTES": 1
            },
            "realfc": {
                "supported": 0,
                "refuted": 1,
                "neutral": 2
            }}


## Profiling
# from line_profiler_pycharm import profile

def get_hardem_prob(steps, start_after, interpolate_until, **kwargs):
    # DEBUG = True
    # if DEBUG:
    #     logger.info("DEBUGGING ACTIVE!!!")
    #     return 1.
    s = steps - start_after
    e = steps - interpolate_until
    if s < 0:
        return 0.0
    if e > 0:
        return 0.95

    interp_steps = interpolate_until - start_after
    return s / interp_steps


class LRMVerifierTrainer:
    def __init__(self, config: dict, global_rank, local_rank, seed):
        self.config = config
        self.global_rank = global_rank
        self.seed = seed
        self.DISTRIBUTED_RESULT_FN = f"lrmreranker_result_S{self.seed}_r.pkl"
        self.best_score = config["save_threshold"]
        # Assuming DebertaV2 is used
        if "deberta" in config["verifier_tokenizer_type"] \
                and self.config.get("log_results", False) or \
                self.config.get("eval_interpretability", False) or \
                self.config.get("eval_interp_during_training", False):
            logger.info("Loading ours DebertaV2TokenizerFast...")
            # Deberta doesn't have fast tokenizer, so we made one
            from .tokenizer.tokenization_deberta_v2_fast import DebertaV2TokenizerFast
            self.tokenizer = init_tokenizer(config, tokenizer_class=DebertaV2TokenizerFast)
        else:
            self.tokenizer = init_tokenizer(config)
        self.distributed = False
        self.last_ckpt_name = ""
        self.local_rank = local_rank
        self.torch_device = local_rank if type(local_rank) != int else torch.device(local_rank)

        self.db = self.getdb()
        self.no_improvement_rounds = 0

        if self.config.get("eval_interpretability", False) or \
                self.config.get("eval_interp_during_training", False):
            self.simpletokenizer = SimpleTokenizer()

    def getdb(self):
        return PassageDB(db_path=self.config['pass_database']) if 'pass_database' in self.config else None

    def get_random_negatives(self, gt_indices, sentence_logits, total_samples, negative_indices=None):
        if negative_indices is None:
            negative_indices = []
        max_its = 10000
        current_its = 0  # loop prevention in special cases, where there is not enough negatives
        while len(negative_indices) < total_samples:
            r = randint(0, len(sentence_logits) - 1)
            current_its += 1
            if current_its > max_its:
                logger.warning("Loop prevention activated!")
                break
            if r not in gt_indices and r not in negative_indices:
                negative_indices.append(r)
        return negative_indices

    def get_nongolden_passage_negatives(self, gt_indices, relevant_passage_indices,
                                        sentence_logits, sentences_per_passage, total_samples,
                                        negative_indices=None, pad_with_random_negatives=True):
        if negative_indices is None:
            negative_indices = []
        # get indices for sentences in gt passages
        only_goldpassage_mask = []
        for i, l in enumerate(sentences_per_passage):
            only_goldpassage_mask += [i in relevant_passage_indices] * l
        non_gt_passages_indices = [x for x, t in zip(range(len(sentence_logits)), only_goldpassage_mask) if not t]

        # remove indices corresponding to gt sentences
        non_golden_sentences_from_nongolden_passages_indices = list(
            set(non_gt_passages_indices) - set(gt_indices))
        # if we need more negatives, than there are in the golden passages, sample at random from golden passages too
        if len(non_golden_sentences_from_nongolden_passages_indices) < total_samples and \
                pad_with_random_negatives:
            negative_indices += non_golden_sentences_from_nongolden_passages_indices
            max_its = 10000
            current_its = 0  # loop prevention in special cases, where there is not enough negatives
            while len(negative_indices) < total_samples:
                r = randint(0, len(sentence_logits) - 1)
                current_its += 1
                if current_its > max_its:
                    logger.warning("Loop prevention activated!")
                    break
                if r not in gt_indices and r not in negative_indices:
                    negative_indices.append(r)
        else:
            shuffle(non_golden_sentences_from_nongolden_passages_indices)
            negative_indices += non_golden_sentences_from_nongolden_passages_indices[:total_samples]
        return negative_indices

    def get_golden_passage_negatives(self, excluded_indices, relevant_passage_indices,
                                     sentence_logits, sentences_per_passage, total_samples,
                                     negative_indices=None, pad_with_random_negatives=True):
        if negative_indices is None:
            negative_indices = []
        # get indices for sentences in gt passages
        only_goldpassage_mask = []
        for i, l in enumerate(sentences_per_passage):
            only_goldpassage_mask += [i in relevant_passage_indices] * l
        gt_passages_indices = [x for x, t in zip(range(len(sentence_logits)), only_goldpassage_mask) if t]

        # remove indices corresponding to gt sentences
        non_golden_sentences_from_golden_passages_indices = list(
            set(gt_passages_indices) - set(excluded_indices))
        # if we need more negatives, than there are in the golden passages, sample at random from non-golden passages too
        if len(non_golden_sentences_from_golden_passages_indices) < total_samples and \
                pad_with_random_negatives:
            negative_indices += non_golden_sentences_from_golden_passages_indices
            max_its = 10000
            current_its = 0  # loop prevention in special cases, where there is not enough negatives
            while len(negative_indices) < total_samples:
                r = randint(0, len(sentence_logits) - 1)
                current_its += 1
                if current_its > max_its:
                    logger.warning("Loop prevention activated!")
                    break
                if r not in excluded_indices and r not in negative_indices:
                    negative_indices.append(r)
        else:
            shuffle(non_golden_sentences_from_golden_passages_indices)
            negative_indices += non_golden_sentences_from_golden_passages_indices[:total_samples]
        return negative_indices

    @staticmethod
    def load_distributed_result(seed=None, distributed_result_fn=None):
        if distributed_result_fn is None:
            distributed_result_fn = f"lrmreranker_result_S{seed}_r.pkl"
        logger.info(f"Loading result from {distributed_result_fn}")
        with open(distributed_result_fn, "rb") as f:
            model_path, validation_accuracy = pickle.load(f)
        os.remove(distributed_result_fn)
        return model_path, validation_accuracy

    def fit(self):
        config = self.config

        logger.debug(json.dumps(config, indent=4, sort_keys=True))

        data = self.get_data(config)
        if data[0] is not None:
            train, val = data
        else:
            train, val = None, data[-1]

        if self.config.get("eval_interp_during_training", False):
            val, val_intp = val
            logger.info(f"TLR-FEVER examples: {len(val_intp)}")

        if not config["test_only"]:
            logger.info(f"Training data examples:{len(train)}")

        logger.info(f"Validation data examples:{len(val)}")

        train_iter = None
        if train is not None:
            train_iter = DataLoader(train, batch_size=1, shuffle=False, pin_memory=True,
                                    collate_fn=FEVERLRMVerifierDataset.
                                    create_collate_fn(pad_t=self.tokenizer.pad_token_id))
        val_iter = DataLoader(val, batch_size=1, shuffle=False, pin_memory=True,
                              collate_fn=FEVERLRMVerifierDataset.
                              create_collate_fn(pad_t=self.tokenizer.pad_token_id))
        if self.config.get("eval_interp_during_training", False):
            val_iter = [val_iter]
            val_iter.append(DataLoader(val_intp, batch_size=1, shuffle=False, pin_memory=True,
                                       collate_fn=FEVERLRMVerifierDataset.
                                       create_collate_fn(pad_t=self.tokenizer.pad_token_id)))
        logger.info("Loading model...")

        model, optimizer, scheduler = self.load_model_optimizer_scheduler()
        if not config["test_only"]:
            start_time = time.time()
            try:
                it = 0
                while get_model(model).training_steps < self.config["max_steps"] and \
                        not self.no_improvement_rounds > self.config.get("patience", 9_999):
                    logger.info(f"Epoch {it}")
                    train_loss = self.train_epoch(model=model,
                                                  data_iter=train_iter,
                                                  val_iter=val_iter,
                                                  optimizer=optimizer,
                                                  lr_scheduler=scheduler)
                    logger.info(f"Training loss: {train_loss:.5f}")
                    it += 1

            except KeyboardInterrupt:
                logger.info('-' * 120)
                logger.info('Exit from training early.')
            finally:
                logger.info(f'Finished after {(time.time() - start_time) / 60} minutes.')
                logger.info(f'Best performance: {self.best_score}')
                logger.info(f"Last checkpoint name: {self.last_ckpt_name}")
        else:
            if self.config.get("eval_interpretability", False):
                if self.config.get("word_level_evidence", False):  # backward compatibility on the set v0.1
                    score = self.validate_interpretability_FEVER(model, train_iter)
                else:
                    score = self.validate_interpretability_FEVER(model, val_iter)
            else:
                score = self.validate(model, val_iter, log_results=self.config.get("log_results", False))
            logger.info(f"Evaluation best score: {score}")
        logger.info("#" * 50)
        return self.best_score, self.last_ckpt_name

    def load_model_optimizer_scheduler(self):
        optimizer = scheduler = None
        if self.config.get("train_masker", False):
            trained_model = torch.load(self.config["train_masker"]["pretrained_model"], map_location=self.torch_device)
            pretrained_model = self.init_model(self.config, self.torch_device, trained_model=trained_model)
            del trained_model
            # freeze pretrained model
            for p in pretrained_model.parameters():
                p.requires_grad = False

            trained_masker = None
            if "trained_masker" in self.config["train_masker"]:
                trained_masker = torch.load(self.config["train_masker"]["trained_masker"],
                                            map_location=self.torch_device)
            model = self.init_masker_model(self.config, self.torch_device, trained_model=trained_masker,
                                           masked_model=pretrained_model.module)
        elif self.config.get("resume_training", False) or self.config.get("pre_initialize", False):
            if self.config.get("resume_training", False):
                logger.info("Resuming training...")
            if not "resume_checkpoint" in self.config:
                self.config["resume_checkpoint"] = self.config["pretrained_reader_model"]

            trained_model = torch.load(self.config["resume_checkpoint"], map_location=self.torch_device)
            model = self.init_model(self.config, self.torch_device, trained_model=trained_model)
            del trained_model
        else:
            if self.config["test_only"]:
                trained_model = torch.load(self.config["model"], map_location=self.torch_device)
                model = self.init_model(self.config, self.torch_device, trained_model=trained_model)
                del trained_model
            else:
                model = self.init_model(self.config, self.torch_device)
                logger.info(f"Model has {count_parameters(model)} trainable parameters")
        logger.info(f"Trainable parameter checksum: {sum_parameters(model)}")
        param_sizes, param_shapes = report_parameters(model)
        param_sizes = "\n'".join(str(param_sizes).split(", '"))
        param_shapes = "\n'".join(str(param_shapes).split(", '"))
        logger.debug(f"Model structure:\n{param_sizes}\n{param_shapes}\n")
        # return
        if not self.config["test_only"]:
            # Init optimizer
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if
                               not any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": self.config["weight_decay"],
                },
                {"params": [p for n, p in model.named_parameters() if
                            any(nd in n for nd in no_decay) and p.requires_grad],
                 "weight_decay": 0.0},
            ]
            if self.config["optimizer"] == "adamw":
                optimizer = AdamW(optimizer_grouped_parameters,
                                  lr=self.config["learning_rate"],
                                  eps=self.config["adam_eps"])
            elif self.config["optimizer"] == "adam":
                optimizer = Adam(optimizer_grouped_parameters,
                                 lr=self.config["learning_rate"],
                                 eps=self.config["adam_eps"])
            else:
                raise ValueError("Unsupported optimizer")

            if self.config.get("resume_checkpoint", False) and not self.config.get("pre_initialize", False):
                optimizer.load_state_dict(get_model(model).optimizer_state_dict)

            # Init scheduler
            if "warmup_steps" in self.config:
                self.config["scheduler_warmup_steps"] = self.config["warmup_steps"]
            if "warmup_proportion" in self.config:
                self.config["scheduler_warmup_proportion"] = self.config["warmup_proportion"]
            if self.config.get("scheduler", None) and \
                    ("scheduler_warmup_steps" in self.config or "scheduler_warmup_proportion" in self.config):
                logger.info("Scheduler active!!!")
                t_total = self.config["max_steps"]
                warmup_steps = round(self.config["scheduler_warmup_proportion"] * t_total) \
                    if "scheduler_warmup_proportion" in self.config else \
                    self.config["scheduler_warmup_steps"]
                scheduler = self.init_scheduler(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=t_total,
                    last_step=get_model(model).training_steps - 1
                )
                logger.info(f"Scheduler: warmup steps: {warmup_steps}, total_steps: {t_total}")
            else:
                scheduler = None
        return model, optimizer, scheduler

    def init_model(self, config, torch_device, trained_model=None):
        model = TransformerClassifier(config).to(torch_device)
        logger.info(f"Resizing token embeddings to length {len(self.tokenizer)}")
        model.transformer.resize_token_embeddings(len(self.tokenizer))

        if trained_model is not None:
            model.load_state_dict(trained_model.state_dict())
            if not config.get("pre_initialize", False):
                model.training_steps = trained_model.training_steps
                model.config = self.config
                if hasattr(trained_model, "optimizer_state_dict"):
                    model.optimizer_state_dict = trained_model.optimizer_state_dict
        if self.config.get("bf16", False):
            model = model.to(dtype=torch.bfloat16)
        return self.make_parallel(model)

    def init_masker_model(self, config, torch_device, trained_model=None, masked_model=None):
        config = copy.deepcopy(config)
        del config["mh_sent_layers"]
        del config["mh_sent_residual"]
        del config["mh_w_sent_tokens"]
        model = TransformerMasker(config).to(torch_device)
        logger.info(f"Resizing token embeddings to length {len(self.tokenizer)}")
        model.transformer.resize_token_embeddings(len(self.tokenizer))

        model.masked_model = masked_model
        if trained_model is not None:
            model.load_state_dict(trained_model.state_dict())
            model.training_steps = trained_model.training_steps
            model.config = config
            if hasattr(trained_model, "optimizer_state_dict"):
                model.optimizer_state_dict = trained_model.optimizer_state_dict
        model.mask_token_id = self.tokenizer.mask_token_id
        sentence_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sentence_special_token)
        model.sentence_token_id = sentence_token_id
        return self.make_parallel(model)

    def get_data(self, config):
        if self.distributed:
            distributed_settings = {
                "rank": self.global_rank,
                "world_size": self.world_size,
            }
        else:
            distributed_settings = None
        train, val = None, None
        eval_int = config["eval_word_level_evidence"] if (config.get('eval_interpretability', False)
                                                          and config.get("eval_word_level_evidence",
                                                                         False)) \
            else False
        if eval_int:
            assert config["dataset"] in ["fever"]  # Word level evidence is only supported for FEVER"

        cheatonval = config.get("cheat_on_val", False) or eval_int

        if config["dataset"] in ["fever"]:
            extra_context_size = None
            if config.get('expand_entities_in_retrieval', None):
                extra_context_size = config["expand_entities_in_retrieval"]["extra_context_size"]
            if config.get("prediction_only", False):
                test = FEVERLRMVerifierDataset(config["test_data"], tokenizer=self.tokenizer,
                                               database=self.db,
                                               transformer=config["verifier_transformer_type"],
                                               cache_dir=config["data_cache_dir"],
                                               max_len=config.get("verifier_max_input_length", None),
                                               shuffle=config.get("shuffle_validation_set", False),
                                               context_length=config["context_length"],
                                               is_training=False,
                                               skip_NEI=config.get('reranking_only', False),
                                               block_size=config["block_size"],
                                               distributed_settings=distributed_settings,
                                               expand_ret=config.get('expand_entities_in_retrieval', None),
                                               extra_context_size=extra_context_size)
                return test
            else:
                val = FEVERLRMVerifierDataset(config["val_data"], tokenizer=self.tokenizer,
                                              database=self.db,
                                              transformer=config["verifier_transformer_type"],
                                              cache_dir=config["data_cache_dir"],
                                              max_len=config.get("verifier_max_input_length", None),
                                              shuffle=config.get("shuffle_validation_set", False),
                                              context_length=config["context_length"],
                                              is_training=False,
                                              cheat_on_val=cheatonval,
                                              skip_NEI=config.get('reranking_only', False),
                                              block_size=config["block_size"],
                                              distributed_settings=distributed_settings,
                                              expand_ret=config.get('expand_entities_in_retrieval', None),
                                              extra_context_size=extra_context_size,
                                              eval_interpretability=eval_int)
                if config.get("eval_interp_during_training", False):
                    assert not eval_int
                    val = [val]
                    val.append(FEVERLRMVerifierDataset(config["val_data"], tokenizer=self.tokenizer,
                                                       database=self.db,
                                                       transformer=config["verifier_transformer_type"],
                                                       cache_dir=config["data_cache_dir"],
                                                       max_len=config.get("verifier_max_input_length", None),
                                                       shuffle=config.get("shuffle_validation_set", False),
                                                       context_length=config["context_length"],
                                                       is_training=False,
                                                       cheat_on_val=True,
                                                       skip_NEI=config.get('reranking_only', False),
                                                       block_size=config["block_size"],
                                                       distributed_settings=distributed_settings,
                                                       expand_ret=config.get('expand_entities_in_retrieval', None),
                                                       extra_context_size=extra_context_size,
                                                       eval_interpretability=config['eval_word_level_evidence']))

                if not config["test_only"] or (
                        config.get("eval_interpretability", False) and config.get("word_level_evidence",
                                                                                  False)):
                    train = FEVERLRMVerifierDataset(config["train_data"], tokenizer=self.tokenizer,
                                                    database=self.db,
                                                    transformer=config["verifier_transformer_type"],
                                                    cache_dir=config["data_cache_dir"],
                                                    max_len=config["verifier_max_input_length"],
                                                    context_length=config["context_length"],
                                                    include_golden_passages=config[
                                                        "make_sure_golden_passages_in_training"],
                                                    is_training=True,
                                                    randomize_context_lengths=config.get(
                                                        "randomize_context_lengths", False),
                                                    shuffle=True,
                                                    skip_NEI=config.get('reranking_only',
                                                                        False) or config.get(
                                                        "train_masker"),
                                                    jiangetal_sup=config.get("jiangetal_sup", None),
                                                    block_size=config["block_size"],
                                                    distributed_settings=distributed_settings,
                                                    expand_ret=config.get('expand_entities_in_retrieval', None),
                                                    extra_context_size=extra_context_size,
                                                    eval_interpretability=config["word_level_evidence"]
                                                    if config.get('eval_interpretability', False) and
                                                       config.get("word_level_evidence", False) else False)
            if config.get("word_level_evidence", False):
                with jsonlines.open(config['word_level_evidence'], "r") as reader:
                    self.word_level_evidence = {e['id']: e for e in reader}
            elif config.get("eval_word_level_evidence", False):
                with jsonlines.open(config['eval_word_level_evidence'], "r") as reader:
                    self.word_level_evidence = {e['id']: e for e in reader}
        elif config["dataset"] in ["hover"]:
            val = HoverLRMVerifierDataset(config["val_data"], tokenizer=self.tokenizer,
                                          database=self.db,
                                          transformer=config["verifier_transformer_type"],
                                          cache_dir=config["data_cache_dir"],
                                          max_len=config.get("verifier_max_input_length", None),
                                          shuffle=config.get("shuffle_validation_set", False),
                                          context_length=config["context_length"],
                                          is_training=False,
                                          cheat_on_val=cheatonval,
                                          block_size=config["block_size"],
                                          distributed_settings=distributed_settings,
                                          eval_interpretability=eval_int)
            if not config["test_only"] or (
                    config.get("eval_interpretability", False) and config.get("word_level_evidence",
                                                                              False)):
                train = HoverLRMVerifierDataset(config["train_data"], tokenizer=self.tokenizer,
                                                database=self.db,
                                                transformer=config["verifier_transformer_type"],
                                                cache_dir=config["data_cache_dir"],
                                                max_len=config.get("verifier_max_input_length", None),
                                                shuffle=config.get("shuffle_validation_set", False),
                                                context_length=config["context_length"],
                                                is_training=True,
                                                khattab_like=config.get("sentence_relevance_negative_ss",
                                                                        "") == "khattab_like",
                                                block_size=config["block_size"],
                                                distributed_settings=distributed_settings,
                                                eval_interpretability=eval_int)
        elif config["dataset"] in ["faviq"]:
            val = FaviqLRMVerifierDataset(config["val_data"], tokenizer=self.tokenizer,
                                          transformer=config["verifier_transformer_type"],
                                          cache_dir=config["data_cache_dir"],
                                          max_len=config.get("verifier_max_input_length", None),
                                          shuffle=config.get("shuffle_validation_set", False),
                                          is_training=False,
                                          block_size=config["block_size"],
                                          distributed_settings=distributed_settings)
            if not config["test_only"] or (
                    config.get("eval_interpretability", False) and config.get("word_level_evidence",
                                                                              False)):
                train = FaviqLRMVerifierDataset(config["train_data"], tokenizer=self.tokenizer,
                                                transformer=config["verifier_transformer_type"],
                                                cache_dir=config["data_cache_dir"],
                                                max_len=config.get("verifier_max_input_length", None),
                                                shuffle=config.get("shuffle_validation_set", False),
                                                is_training=True,
                                                randomize_context_lengths=config.get(
                                                    "randomize_context_lengths", False),
                                                block_size=config["block_size"],
                                                distributed_settings=distributed_settings)
        elif config["dataset"] in ["realfc"]:
            val = RealFCLRMVerifierDataset(config["val_data"], tokenizer=self.tokenizer,
                                           transformer=config["verifier_transformer_type"],
                                           cache_dir=config["data_cache_dir"],
                                           max_len=config.get("verifier_max_input_length", None),
                                           shuffle=config.get("shuffle_validation_set", False),
                                           is_training=False,
                                           block_size=config["block_size"],
                                           distributed_settings=distributed_settings)
            if not config["test_only"]:
                train = RealFCLRMVerifierDataset(config["train_data"], tokenizer=self.tokenizer,
                                                 transformer=config["verifier_transformer_type"],
                                                 cache_dir=config["data_cache_dir"],
                                                 max_len=config.get("verifier_max_input_length", None),
                                                 shuffle=config.get("shuffle_validation_set", False),
                                                 is_training=True,
                                                 block_size=config["block_size"],
                                                 distributed_settings=distributed_settings)
        else:
            raise NotImplemented(f"Unknown dataset {config['dataset']}")
        return train, val

    @torch.no_grad()
    def predict(self):
        config = self.config
        logger.debug(json.dumps(config, indent=4, sort_keys=True))
        test = self.get_data(config)
        logger.info(f"Test data examples:{len(test)}")

        test_iter = DataLoader(test, batch_size=1, shuffle=False, pin_memory=True,
                               collate_fn=FEVERLRMVerifierDataset.
                               create_collate_fn(pad_t=self.tokenizer.pad_token_id))
        logger.info("Loading model...")
        trained_model = torch.load(config["model"], map_location=self.torch_device)
        model = self.init_model(config, self.torch_device, trained_model=trained_model)
        del trained_model
        logger.info(f"Total parameters {count_parameters(model)}")
        logger.info(f"Trainable parameter checksum: {sum_parameters(model)}")
        param_sizes, param_shapes = report_parameters(model)
        param_sizes = "\n'".join(str(param_sizes).split(", '"))
        param_shapes = "\n'".join(str(param_shapes).split(", '"))
        logger.debug(f"Model structure:\n{param_sizes}\n{param_shapes}\n")

        outfile = f".predictions/predictions_{uuid.uuid4().hex}.jsonl"
        logger.info(f"Generating predictions into: {outfile}")
        self._predict(model, test_iter, outfile=outfile)

    def init_scheduler(self, optimizer: Optimizer, num_warmup_steps: int,
                       num_training_steps: int, last_step: int = -1) -> LambdaLR:
        """
        Initialization of lr scheduler.

        :param last_step:
        :param num_training_steps:
        :param optimizer: The optimizer that is used for the training.
        :type optimizer: Optimizer
        :return: Created scheduler.
        :rtype: LambdaLR
        """
        if last_step > 0:
            logger.info(f"Setting scheduler step to {last_step}")
            # We need initial_lr, because scheduler demands it.
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])

        if self.config["scheduler"] == "linear":
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                last_epoch=last_step)
        elif self.config["scheduler"] == "cosine":
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=0.5,
                last_epoch=last_step)
        elif self.config["scheduler"] == "constant":
            scheduler = transformers.get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                last_epoch=last_step)
        else:
            scheduler = None

        return scheduler

    def train_epoch(self,
                    model: TransformerClassifier,
                    data_iter,
                    val_iter,
                    optimizer: Optimizer,
                    lr_scheduler: LambdaLR):
        #  Training flags
        model.train()
        # Make sure parameters are zero
        optimizer.zero_grad()

        # Determine update ratio, e.g. if true_batch_size = 32 and batch_size=8, then
        # gradients should be updated  every 4th iteration (except for last update!)
        update_ratio = self.config["true_batch_size"] // self.config["batch_size"]
        assert self.config["true_batch_size"] % self.config["batch_size"] == 0
        updated = False
        adjusted_for_last_update = False  # In last update, the ba tch size is adjusted to whats left

        # Calculate total number of updates per epoch
        total = ceil(len(data_iter.dataset) / data_iter.batch_size)

        # For progressive  training loss  reporting
        total_losses = []
        losses_per_update = []

        masked_tokens = []

        # If we use fp16, gradients must be scaled
        grad_scaler = None
        if self.config["fp16"]:
            grad_scaler = torch.cuda.amp.GradScaler()

        it = tqdm(data_iter, total=total)
        iteration = 0
        for src, src_type_ids, src_mask, label, metadata in it:
            # Move to gpu
            src, src_mask = src.to(self.torch_device), src_mask.to(self.torch_device)
            if src_type_ids is not None:
                src_type_ids.to(self.torch_device)
            # if metadata[0]['id'] not in self.word_level_evidence:
            #     continue
            iteration += 1
            updated = False
            assert len(src) == 1  # more  than 1 example per batch is unsupported

            src = src[0]
            src_mask = src_mask[0]
            if src_type_ids is not None:
                src_type_ids = src_type_ids[0]
            # assert self.tokenizer.pad_token_id not in src[src_mask.bool()].view(-1).tolist()

            src_shapes = src.shape
            src_mask_shapes = src_mask.shape
            try:
                # Useful for debugging
                # inps = [" ".join(self.tokenizer.convert_ids_to_tokens(inp)) for inp in src]

                # Adjust update ratio for last update if needed
                if (total - iteration) < update_ratio and len(losses_per_update) == 0 and not adjusted_for_last_update:
                    logger.debug(f"I{iteration}_S{get_model(model).training_steps},Adjusting for last update!")
                    update_ratio = (total - iteration)
                    adjusted_for_last_update = True

                # If DDP is active, synchronize gradients only in update step!
                # Do not do this in the last minibatch, if dataset is not divisible by minibatch (and thus there was no adjustment)
                if self.distributed and \
                        not (len(losses_per_update) + 1 == update_ratio) \
                        and not adjusted_for_last_update:
                    with model.no_sync():
                        loss, validation_outputs = self.forward_pass(src, src_type_ids, src_mask, label, metadata,
                                                                     model)
                        if (loss, validation_outputs) == (None, None):
                            continue
                        loss /= update_ratio
                        grad_scaler.scale(loss).backward() if self.config["fp16"] else loss.backward()
                        # logger.debug(
                        #     f"R:{self.global_rank}: no_sync! (LPU:{len(losses_per_update)}, UR:{update_ratio})")
                else:
                    loss, validation_outputs = self.forward_pass(src, src_type_ids, src_mask, label, metadata, model)
                    if (loss, validation_outputs) == (None, None):
                        continue
                    loss /= update_ratio
                    grad_scaler.scale(loss).backward() if self.config["fp16"] else loss.backward()
                    # logger.debug(f"R:{self.global_rank}: synced! (LPU:{len(losses_per_update)}, UR:{update_ratio})")

                # record losses to list
                losses_per_update.append(loss.item())

                if "mask_weights" in validation_outputs:
                    masked_tokens.append(validation_outputs['mask_weights'].sum().item())

                if len(losses_per_update) == update_ratio and not adjusted_for_last_update:
                    # check that the model is in training mode
                    assert model.training

                    # grad clipping should be applied to unscaled gradients
                    if self.config["fp16"]:
                        # Unscales the gradients of optimizer's assigned params in-place
                        grad_scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                                   self.config["max_grad_norm"])
                    # compute training loss
                    loss_per_update = sum(losses_per_update)

                    # if we train masker, get avg amount of masked tokens
                    if masked_tokens:
                        average_masked_tokens = sum(masked_tokens) / len(masked_tokens)
                        masked_tokens = []
                        logger.debug(
                            f"Loss: {loss_per_update}, EID: {metadata[0]['id']},"
                            f" training {model.training}, masked_avg {average_masked_tokens:.2f}")
                    else:
                        logger.debug(f"Loss: {loss_per_update}, EID: {metadata[0]['id']}")
                    if math.isnan(loss_per_update):
                        logger.debug(f"Losses_pu" + str(losses_per_update))

                    total_losses += losses_per_update
                    losses_per_update = []

                    if self.config["fp16"]:
                        # Unscales gradients and calls
                        # or skips optimizer.step()
                        grad_scaler.step(optimizer)
                        # Updates the scale for next iteration
                        grad_scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    get_model(model).training_steps += 1
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    updated = True

                    it.set_description(f"Steps: {get_model(model).training_steps}, "
                                       f"Training loss: {loss_per_update:.5f}")

                    # Validate after every validate_after_steps steps
                    if get_model(model).training_steps > 1 and \
                            get_model(model).training_steps % self.config["validate_after_steps"] == 0:
                        best_score, improvement = self.validate(model, val_iter,
                                                                optimizer_dict=optimizer.state_dict())
                        model = model.train()

                        self.no_improvement_rounds = 0 if improvement else self.no_improvement_rounds + 1

                    # Exit if maximal number of steps is reached, or patience is surpassed
                    if get_model(model).training_steps == self.config["max_steps"] or \
                            self.no_improvement_rounds > self.config.get("patience", 9_999):
                        break
            # Catch out-of-memory errors
            except RuntimeError as e:
                if "CUDA out of memory." in str(e):
                    torch.cuda.empty_cache()
                    logger.error(str(e))
                    logger.error("OOM detected, emptying cache...")
                    logger.error(f"src_shape {src_shapes}\n"
                                 f"src_mask_shape{src_mask_shapes}\n"
                                 )
                    time.sleep(3)
                else:
                    raise e
        if not updated:
            # logger.debug(f"I{iteration}_S{get_model(model).training_steps}, Doing last step update R{self.rank}"
            #               f"from {len(losses_per_update)} losses")
            # check that the model is in training mode
            assert model.training
            # Do the last step if needed
            if self.config["fp16"]:
                grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                           self.config["max_grad_norm"])
            if self.config["fp16"]:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            get_model(model).training_steps += 1
            if lr_scheduler is not None:
                lr_scheduler.step()
            losses_per_update = []

        # Validate after epoch
        self.validate(model, val_iter,
                      optimizer_dict=optimizer.state_dict())
        return sum(total_losses) / len(total_losses)

    def forward_pass(self, *args, **kwargs):
        """
        FP16 wrapper
        """
        if self.config["fp16"]:
            with torch.cuda.amp.autocast():
                return self._forward_pass(*args, **kwargs)
        elif self.config.get("bf16", False):
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                return self._forward_pass(*args, **kwargs)
        else:
            return self._forward_pass(*args, **kwargs)

    def _forward_pass(self, src, src_type_ids, src_mask, label=None, metadata=None, model=None, validation=False):

        sentence_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sentence_special_token)

        passage_mask = src_mask.bool().view(-1)
        fid_shaped_src = src.view(-1)[passage_mask]
        # Variant 1, just take sentence tokens
        sentence_tokens_mask = (fid_shaped_src == sentence_token_id)

        input_parameters = {
            "input_ids": src,
            "token_type_ids": src_type_ids if src_type_ids != [] else None,
            "attention_mask": src_mask
        }
        if self.config.get("mh_w_sent_tokens", False):
            input_parameters["sentence_tokens_mask"] = sentence_tokens_mask
        if self.config.get("train_masker", False):
            input_parameters['evidence_mask'] = self.get_evidence_mask(src, metadata,
                                                                       mask_nongts=self.config["train_masker"].get(
                                                                           "mask_only_gt", False))
        gt_dict = gt_dicts[self.config["dataset"]]
        if "evidence_mask" in input_parameters:
            block_mask = (input_parameters['evidence_mask'].sum(-1) > 0)
            if not validation and block_mask.sum() == 0:  # this can rarely happen!
                return None, None

        outputs = model(**input_parameters)

        if self.config.get('prediction_only', False):
            allcls_inputwise_logits = outputs
            # as this is not FID, we need to mask out the non-pad logits!
            allcls_inputwise_logits = allcls_inputwise_logits.view(-1, allcls_inputwise_logits.shape[-1])[
                passage_mask].unsqueeze(0)
            _marg_logprobs, _ = self.get_marg_logprobs(src, src_mask, metadata[0],
                                                       allcls_inputwise_logits)
            if self.config["dataset"] in ["hover", "faviq"]:
                marg_logprobs = torch.stack((_marg_logprobs[0], torch.logsumexp(_marg_logprobs[1:], 0)))
            else:
                marg_logprobs = _marg_logprobs
            validation_outputs = {
                "marg_logprobs": marg_logprobs,
                "output_logits": allcls_inputwise_logits,
            }
            return None, validation_outputs

        passage_mask = src_mask.bool().view(-1)
        wl_loss_activated = self.config.get("word_level_evidence", False) and \
                            metadata[0]['id'] in self.word_level_evidence and \
                            not self.config.get("eval_interpretability", False)
        labels = torch.LongTensor([gt_dict[label[0]]]).to(self.torch_device)
        if self.config.get("train_masker", False):
            allcls_inputwise_logits = outputs['outputs']
            if allcls_inputwise_logits is None:
                validation_outputs = {
                    "mask_weights": outputs['mask'],
                }
                return -1, validation_outputs
            # as this is not FID, we need to mask out the non-pad logits!
            allcls_inputwise_logits = allcls_inputwise_logits.view(-1, allcls_inputwise_logits.shape[-1])[
                passage_mask].unsqueeze(0)
            marg_logprobs, _ = self.get_marg_logprobs(src, src_mask, metadata[0], allcls_inputwise_logits)
            loss = - marg_logprobs[gt_dict["NOT ENOUGH INFO"]]
            total = input_parameters['evidence_mask'].sum()

            sparsity_loss = outputs['mask'].sum() / total  # computed on "sampled probs" of mask class
            loss += sparsity_loss * self.config["train_masker"]["sparsity_weight"]

            validation_outputs = {
                "marg_logprobs": marg_logprobs.detach(),
                "marg_labels": labels.detach(),
                "mask_weights": outputs['mask'],
            }
            return loss, validation_outputs
        else:
            if self.config.get("dump_attention_matrices", False):
                allcls_inputwise_logits, mhweights = outputs
            else:
                allcls_inputwise_logits = outputs

            if not self.config.get("mh_clustered", False):
                # as this is not FID, we need to mask out the non-pad logits!
                allcls_inputwise_logits = allcls_inputwise_logits.view(-1, allcls_inputwise_logits.shape[-1])[
                    passage_mask].unsqueeze(0)
            losses = self.get_endpoints_loss_from_marglogits(src, src_mask,
                                                             metadata[0],
                                                             allcls_inputwise_logits,
                                                             steps=get_model(model).training_steps,
                                                             marg_labels=labels,
                                                             validation=validation)
            sent_loss = losses["sent_loss"]
            wl_loss = losses.get("wl_loss", 0.)
            L_reg_loss = losses.get("lreg_loss", 0.)
            L_reg_sent_loss = losses.get("lreg_sentloss", 0.)
            _marg_logprobs, _ = self.get_marg_logprobs(src, src_mask, metadata[0],
                                                       allcls_inputwise_logits)
            if self.config["dataset"] in ["hover", "faviq"]:
                marg_logprobs = torch.stack((_marg_logprobs[0], torch.logsumexp(_marg_logprobs[1:], 0)))
            else:
                marg_logprobs = _marg_logprobs

            output_logits = allcls_inputwise_logits
        # self.get_marg_logprobs already returns log_probs for every class
        marg_loss = - torch.gather(marg_logprobs.unsqueeze(0), -1, labels.unsqueeze(-1)).mean()

        # FIX ISNAN IN PERWORDDROPOUTING SPECIAL CASES
        if torch.isnan(sent_loss):
            logger.error("NAN RECORDED!!!!SRC:")
            logger.error(f"{str(src.cpu().tolist())}")
        loss = self.config['lossterm_weights']['marg_loss'] * marg_loss + \
               self.config['lossterm_weights']['sent_loss'] * sent_loss

        if not self.config.get('skip_loss_averaging', False):
            loss /= 2

        if wl_loss_activated:
            loss += self.config["wlann_weight"] * wl_loss
        if "perword_laplace" in self.config:
            loss += self.config["perword_laplace"] * L_reg_loss
        if "perword_L2" in self.config:
            loss += self.config["perword_L2"] * L_reg_loss
        if "persentencescoresum_L2" in self.config:
            loss += self.config["persentencescoresum_L2"] * L_reg_sent_loss

        validation_outputs = {
            "marg_logprobs": marg_logprobs,
            "marg_labels": labels,
            "output_logits": output_logits,
            # "sentence_relevance_logprobs": sentence_relevance_logprobs
        }
        if self.config.get("dump_attention_matrices", False):
            validation_outputs["attention_weights"] = mhweights

        balance_weight = 1.
        if "balance_weights" in self.config:
            loss = loss * self.config["balance_weights"][gt_dict[label[0]]]
        return loss, validation_outputs

    def get_marg_logprobs(self, src, src_mask, metadata, all_passage_tokens_class_logits):
        # For special cases, so the torch distributed won't complain
        ZERO = all_passage_tokens_class_logits.sum() * torch.zeros(1).squeeze()
        sentence_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sentence_special_token)
        passage_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.passage_special_token)

        passage_mask = src_mask.bool().view(-1)
        fid_shaped_src = src.view(-1)[passage_mask]

        sentence_tokens_mask = (fid_shaped_src == sentence_special_token)

        if sum(sentence_tokens_mask) == 0:
            return ZERO

        if self.config["dataset"] in ["fever", "faviq", "hover"]:
            # sanity check; there must be the same amount of sentences as in sentence ranges
            total_sentences_in_metadata = len([y for x in metadata['sent_ranges'] for y in x.split("|")])
            assert sum(sentence_tokens_mask) == total_sentences_in_metadata
        elif self.config["dataset"] in ["realfc"]:
            total_sentences_in_metadata = len(metadata['sentence_labels'])
            assert sum(sentence_tokens_mask) == total_sentences_in_metadata

        all_passage_tokens_class_logits = all_passage_tokens_class_logits.squeeze(0)

        # Extract sentence tokens
        unnormalized_logits = self.extract_unnormalized_logits(all_passage_tokens_class_logits,
                                                               fid_shaped_src.tolist(),
                                                               passage_special_token, sentence_special_token)
        # Test
        # unnormalized_logits2 = self.extract_unnormalized_logits_(all_passage_tokens_3class_logits,
        #                                                         fid_shaped_src.tolist(),
        #                                                         passage_special_token, sentence_special_token)
        # assert unnormalized_logits.allclose(unnormalized_logits2)

        """
        It can be proved these two if rows are equivalent 
        (except that first also returns normalized_logits as second param)
        """
        if self.config.get("margclassifier_global_normalize", False) or \
                self.config.get("paper_baseline", False):
            # convert into logprobs and logsumexp
            normalized_logits = F.log_softmax(unnormalized_logits.view(-1), 0).view(unnormalized_logits.shape)
            return torch.logsumexp(normalized_logits, dim=0), normalized_logits
        else:
            # logsumexp logits for each class
            return torch.logsumexp(unnormalized_logits, dim=0) - torch.logsumexp(unnormalized_logits.view(-1), 0), \
                unnormalized_logits

    def get_endpoints_loss_from_marglogits(self, src, src_mask, metadata, all_passage_tokens_logits, steps,
                                           marg_labels=None, validation=False):
        if self.config.get("negative_sentence_supervision", False):
            assert self.config.get("paragraph_supervision", False)
        negative_sampling_strategy = self.config.get("sentence_relevance_negative_ss", "")
        compute_wl_loss = self.config.get("word_level_evidence", False) \
                          and metadata['id'] in self.word_level_evidence and \
                          not self.config.get("eval_interpretability", False)
        # For special cases, so the torch distributed won't complain
        ZERO = all_passage_tokens_logits.sum() * torch.zeros(1).squeeze()
        if self.config["dataset"] in ["fever", "faviq", "hover"]:
            if negative_sampling_strategy == "" or \
                    ('relevant_passage_labels' in metadata and metadata['relevant_passage_labels'] == []):
                return {"sent_loss": ZERO}
        sentence_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sentence_special_token)
        passage_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.passage_special_token)

        passage_mask = src_mask.bool().view(-1)
        fid_shaped_src = src.view(-1)[passage_mask]
        fid_shaped_src_list = fid_shaped_src.tolist()

        sentence_tokens_mask = (fid_shaped_src == sentence_special_token)

        if self.config["dataset"] in ["fever", "faviq", "hover"]:
            total_sentences_in_metadata = len([y for x in metadata['sent_ranges'] for y in x.split("|")])
            assert sum(sentence_tokens_mask) == total_sentences_in_metadata
            # get count of sentences in each passage
            each_passage_sentence_count = [len(x.split("|")) for x in metadata['sent_ranges']]
        elif self.config["dataset"] in ["realfc"]:
            total_sentences_in_metadata = len(metadata['sentence_labels'])
            if not sum(sentence_tokens_mask) == total_sentences_in_metadata:
                print(sum(sentence_tokens_mask))
                print(total_sentences_in_metadata)
                print(metadata)
                print(src)
                print(str([" ".join(self.tokenizer.convert_ids_to_tokens(s)) for s in src]))
            assert sum(sentence_tokens_mask) == total_sentences_in_metadata
            each_passage_sentence_count = [s.tolist().count(sentence_special_token) for s in src]

        all_passage_tokens_logits = all_passage_tokens_logits.squeeze(0)

        if self.config["dataset"] in ["fever", "faviq", "hover"]:
            if "relevant_sentence_labels" in metadata:
                gt_indices = []
                gt_passage_indices = []
                for passage_idx, sent_idx in metadata['relevant_sentence_labels']:
                    assert each_passage_sentence_count[passage_idx] > sent_idx
                    flat_sentence_index = sum(each_passage_sentence_count[:passage_idx]) + sent_idx
                    gt_indices.append(flat_sentence_index)
                    gt_passage_indices.append(passage_idx)
            else:
                gt_passage_indices = metadata['relevant_passage_labels']
                gt_indices = [None] * len(metadata['relevant_passage_labels'])

            if gt_indices == []:
                return {"sent_loss": ZERO}

            gt_passage_indices = set(gt_passage_indices)
            if validation:
                negative_indices = []
                negative_passage_indices = []
            elif negative_sampling_strategy == "jiangetal_sup":
                negative_indices = []
                negative_passage_indices = []
                irrelevant_passage_indices = []
                for passage_idx, sent_idx in metadata['irrelevant_sentence_labels']:
                    assert each_passage_sentence_count[passage_idx] > sent_idx
                    flat_sentence_index = sum(each_passage_sentence_count[:passage_idx]) + sent_idx
                    negative_passage_indices.append(passage_idx)
                    negative_indices.append(flat_sentence_index)
                    irrelevant_passage_indices.append(passage_idx)

                negative_indices = list(set(negative_indices))
                negative_passage_indices = list(set(negative_passage_indices))
            elif negative_sampling_strategy == "random_golden_nongolden_mixture":
                if "relevant_sentence_labels" in metadata:
                    total_sentence_tokens = sum(sentence_tokens_mask).item()
                    possible_negatives = set(range(total_sentence_tokens)) - set(gt_indices)
                    negative_indices = sample(possible_negatives,
                                              k=min([len(gt_indices) * self.config["xtimesneg"],
                                                     len(possible_negatives)]))
                else:
                    total_sentence_tokens = list(range(sum(sentence_tokens_mask).item()))
                    sentences_in_positive_passages = []
                    for p in gt_passage_indices:
                        sentences_in_positive_passages += total_sentence_tokens[
                                                          sum(each_passage_sentence_count[:p]):sum(
                                                              each_passage_sentence_count[:p + 1])]
                    possible_negatives = set(total_sentence_tokens) - set(sentences_in_positive_passages)
                    negative_indices = sample(possible_negatives,
                                              k=min([len(gt_indices) * self.config["xtimesneg"],
                                                     len(possible_negatives)]))
                possible_negative_passages = set(range(len(src))) - set(gt_passage_indices)
                negative_passage_indices = sample(possible_negative_passages, k=min(
                    [len(gt_passage_indices) * self.config["xtimesneg"], len(possible_negative_passages)]))
            elif negative_sampling_strategy == "khattab_like":
                all_gt_passage_sentence_indices = []
                for gt_passage_idx in deduplicate_list(gt_passage_indices):
                    first_s_idx = sum(each_passage_sentence_count[:gt_passage_idx])
                    num_of_sentences = each_passage_sentence_count[gt_passage_idx]
                    all_gt_passage_sentence_indices += list(range(first_s_idx, first_s_idx + num_of_sentences))

                negative_gt_passage_sentence_indices = [i for i in all_gt_passage_sentence_indices if
                                                        i not in gt_indices]

                def get_all_indices(l, item):
                    return [i for i, x in enumerate(l) if x == item]

                negative_lowrankedpassage_passage_indices = [item for irrelevant_title in metadata['irrelevant_titles']
                                                             for item in
                                                             get_all_indices(metadata['titles'], irrelevant_title)]
                # make sure the passage is not GT passage, by accident
                negative_lowrankedpassage_passage_indices = [item for item in negative_lowrankedpassage_passage_indices
                                                             if
                                                             item not in gt_passage_indices]

                negative_lowrankedpassage_sentence_indices = []
                for neg_passage_idx in negative_lowrankedpassage_passage_indices:
                    first_s_idx = sum(each_passage_sentence_count[:neg_passage_idx])
                    num_of_sentences = each_passage_sentence_count[neg_passage_idx]
                    negative_lowrankedpassage_sentence_indices += list(
                        range(first_s_idx, first_s_idx + num_of_sentences))
                negative_indices = deduplicate_list(negative_gt_passage_sentence_indices +
                                                    negative_lowrankedpassage_sentence_indices)
                negative_passage_indices = deduplicate_list(negative_lowrankedpassage_passage_indices)
            else:
                raise ValueError(f"Unsupported negative sampling strategy! {negative_sampling_strategy}")
            if self.config.get("sample_negs_from_golden_passages", False):
                p = self.config["sample_negs_from_golden_passages"]["p"]
                N = self.config["sample_negs_from_golden_passages"]["N"]
                if random() < p:
                    all_gt_passage_sentence_indices = []
                    possible_passage_idx = dict()
                    for passage_idx, sent_idx in metadata['relevant_sentence_labels']:
                        previous_sentences = sum(each_passage_sentence_count[:passage_idx])
                        gt_passage_sent_ids = [previous_sentences + i for i in
                                               range(each_passage_sentence_count[passage_idx])]
                        all_gt_passage_sentence_indices += gt_passage_sent_ids
                        possible_passage_idx[passage_idx] = gt_passage_sent_ids
                    possible_negatives_from_gt_passages = set(all_gt_passage_sentence_indices) - set(gt_indices)
                    negative_indices_from_gt_sentence_passages = sample(possible_negatives_from_gt_passages, k=min(
                        [N, len(possible_negatives_from_gt_passages)]))
                    negative_passage_indices_with_gt_sentences = [passage_id for passage_id in
                                                                  possible_passage_idx.keys()
                                                                  if any(x in possible_passage_idx[passage_id]
                                                                         for x in
                                                                         negative_indices_from_gt_sentence_passages)]
                    negative_indices = list(set(negative_indices + negative_indices_from_gt_sentence_passages))
                    negative_passage_indices = list(
                        set(negative_passage_indices + negative_passage_indices_with_gt_sentences))
            if self.config.get("paragraph_supervision", False):
                if self.config.get("negative_sentence_supervision", False):
                    negative_sentence_indices = negative_indices
                    negative_passage_indices = []
                    assert not self.config.get("paper_baseline", False)
                    # negative logits will be picked from sentence representations, not passage, so let negatives be []

                gt_indices = gt_passage_indices
                negative_indices = negative_passage_indices

            gt_indices = list(gt_indices)
            negative_indices = list(negative_indices)

            considered_indices = gt_indices + negative_indices
        elif self.config["dataset"] in ["realfc"]:
            considered_indices = None
        if "hard_paragraph_supervision" in self.config and not validation:
            p = get_hardem_prob(steps=steps, **self.config['hard_paragraph_supervision'])
            hard_paragraph_supervision = random() < p
        else:
            hard_paragraph_supervision = False
        logprob_collection_args = {
            "all_passage_tokens_logits": all_passage_tokens_logits,
            "fid_shaped_src": fid_shaped_src_list,
            "passage_special_token": passage_special_token,
            "sentence_special_token": sentence_special_token,
            "considered_indices": considered_indices,
            "paragraph_supervision": self.config.get("paragraph_supervision", False),
            "hard_paragraph_supervision": hard_paragraph_supervision,
            "true_hardem": self.config.get("hard_paragraph_supervision", dict()).get("true_hardem", False),
            "stochastic_em": self.config.get("hard_paragraph_supervision", dict()).get("stochastic_em", False),
            "return_perword_logprobs": True,
            "return_perword_logits": True
        }
        provenance_logits, perword_logprobs, perword_logits = self.get_logprobs_per_sentence_from_logits(
            **logprob_collection_args)
        if self.config.get("negative_sentence_supervision", False) and not validation:
            logprob_collection_args["paragraph_supervision"] = False
            logprob_collection_args["hard_paragraph_supervision"] = False
            logprob_collection_args["true_hardem"] = False
            logprob_collection_args["stochastic_em"] = False
            sentence_logits, sentence_perword_logprobs, sentence_perword_logits = self.get_logprobs_per_sentence_from_logits(
                **logprob_collection_args)
        # sentence_logits2 = self.get_logprobs_per_sentence_from_logits_(all_passage_tokens_logits,
        #                                                                fid_shaped_src_list,
        #                                                                passage_special_token,
        #                                                                sentence_special_token)
        # assert sentence_logits.allclose(sentence_logits2)
        assert provenance_logits.shape[0] == total_sentences_in_metadata or \
               (self.config.get("paper_baseline", False) and
                provenance_logits.shape[0] == len(considered_indices) and self.config["dataset"] in ["fever"]) or \
               (self.config.get("paragraph_supervision", False) and provenance_logits.shape[0] == len(src))
        sentences_per_passage = (src == sentence_special_token).sum(1).cpu()
        assert all((torch.IntTensor(each_passage_sentence_count) == sentences_per_passage).tolist())

        # SANITY CHECK!
        SANITY_CHECK = False
        if SANITY_CHECK:
            detokenized_fid_src = self.tokenizer.convert_ids_to_tokens(fid_shaped_src)
            sent_indices = sentence_tokens_mask.nonzero(as_tuple=True)[0]
            gt_sent_indices = sent_indices[gt_indices]
            gt_sent_ranges = []
            for i in range(len(sent_indices)):
                if sent_indices[i] in gt_sent_indices:
                    # from the last <sentence> token to this <sentence> token
                    # in case it is the first, take the sentence starting right after passage_special token
                    start = sent_indices[i - 1].item() + 1 if i > 0 else \
                        detokenized_fid_src.index(self.tokenizer.passage_special_token) + 1
                    end = sent_indices[i].item() + 1
                    # if passage boundary is crossed, the start is still from the last passage, and <claim> will be present in this range
                    if self.tokenizer.claim_special_token in detokenized_fid_src[start:end]:
                        start = start + detokenized_fid_src[start:].index(self.tokenizer.passage_special_token) + 1
                    gt_sent_ranges.append((start, end))
            gt_sentences_decoded = []
            for rng in gt_sent_ranges:
                gt_sentences_decoded.append(
                    " ".join(detokenized_fid_src[rng[0]:rng[1]]))

            claim = detokenized_fid_src[:detokenized_fid_src.index(self.tokenizer.title_special_token)]
            logger.info(f"Claim: {' '.join(claim)}")
            for idx, sentence in enumerate(gt_sentences_decoded):
                logger.info(f"Supporting sentence {idx}: {sentence}")

        # if not validation and not negative_indices:
        #     return ZERO

        if compute_wl_loss:
            """
            This loss uses word-level supervision to boost the word-level logprobs in the model. 
            This loss and supervision were eventually not used in the paper.
            """
            assert not self.config.get("paper_baseline", False)
            wl_loss = self.compute_wordlevel_loss(perword_logprobs, self.word_level_evidence[metadata['id']],
                                                  fid_shaped_src_list, gt_indices, marg_labels)
        L_reg_loss = L_sent_reg_loss = None
        """
        Regularization loss for the word-level logits. L2 was used in the paper.
        """
        if self.config.get("perword_laplace", 0) > 1e-6:
            L_reg_loss = perword_logits[~torch.isinf(perword_logits)] \
                             .abs().sum() / perword_logits.shape[0]
        elif self.config.get("perword_L2", 0) > 1e-6:
            L_reg_loss = perword_logits[~torch.isinf(perword_logits)] \
                             .square().sum() / perword_logits.shape[0]

        if self.config.get("persentencescoresum_L2", 0) > 1e-6:
            """
            L2 prior on sentence-level. SSE turned out to work better, so this was not used within the paper.
            """
            L_sent_reg_loss = self.get_persentencescoresum_L2reg(all_passage_tokens_logits,
                                                                 fid_shaped_src_list,
                                                                 passage_special_token,
                                                                 sentence_special_token)
        if self.config["dataset"] in ["fever", "faviq", "hover"]:
            if not self.config.get("paper_baseline", False):
                positive_logits = provenance_logits[gt_indices]
                negative_logits = provenance_logits[negative_indices]
                if self.config.get("negative_sentence_supervision", False) and not validation:
                    negative_sentence_logprobs = sentence_logits[negative_sentence_indices]
                    negative_logits = negative_sentence_logprobs

                allclass_logits = torch.cat((positive_logits, negative_logits))
                allclass_log_probs = allclass_logits

        if self.config["dataset"] in ["realfc"]:
            """
            Here labels are exhaustive. Every sentence contains supporting, refuting or neutral stance label.
            """
            output_classes = 3
            classdict = {"supporting": 0, "refuting": 1, "neutral": 2}
            allclass_log_probs = provenance_logits
            # push positives UP by maximizing correct class and negatives DOWN by maximizing NEI class
            targets = torch.LongTensor([classdict[l] for l in metadata['sentence_labels']]).to(
                allclass_log_probs.device)

            log_probs_for_loss = torch.gather(allclass_log_probs, -1, targets.unsqueeze(-1))

            # Weighted average, weighting according to # of sentences of that class
            statistics = [0 for _ in range(3)]
            for target in targets.tolist():
                statistics[target] += 1
            statistics = [1. / s if s > 0. else 0. for s in statistics]
            normalizers = torch.FloatTensor([statistics[t] for t in targets.tolist()]).to(allclass_log_probs.device)
            sent_loss = - (log_probs_for_loss * normalizers.unsqueeze(-1)).sum() / len(set(targets.tolist()))

        elif self.config.get("paper_baseline", False):
            # provenance_logits are  globally normalized in this case

            # V1: marginalize over relevant documents and all relevant labels
            if self.config["baseline_version"] == "V1":
                sent_loss = - torch.logsumexp(provenance_logits[:len(gt_indices)].view(-1), 0)

            # V2: marginalize of correct labels
            elif self.config["baseline_version"] == "V2":
                correct_label = marg_labels[0].item()
                sent_loss = - torch.logsumexp(provenance_logits[:len(gt_indices), correct_label].view(-1), 0)

            # V3: maximize correct label of relevant items
            elif self.config["baseline_version"] == "V3":
                correct_label = marg_labels[0].item()
                sent_loss = - torch.mean(provenance_logits[:len(gt_indices), correct_label].view(-1), 0)

            # V4: maximize correct label of
            elif self.config["baseline_version"] == "V4":
                # V4 MULTIPLYING, and PICKING CORRECT CLASS
                # This is not working for joint probability
                correct_label = torch.LongTensor(
                    [marg_labels[0].item()] * len(gt_indices) + [2] * (len(provenance_logits) - len(gt_indices))) \
                    .to(provenance_logits.device)
                gathered_logprobs = torch.gather(provenance_logits, -1, correct_label.unsqueeze(-1))
                sent_loss = - gathered_logprobs.mean()
            else:
                raise ValueError("Unknown baseline version")
        elif self.config.get("no_logsumexp_relevance", False):
            """
            Here, it is assumed that final veracity class is the same as stance of each evidence 
            This if SUPPORT is finally veracity, we assume that every relevant evidence SUPPORTs the claim
            """
            output_classes = 3
            correct_label = marg_labels[0].item()
            assert correct_label in [0, 1]
            # push positives UP by maximizing correct class and negatives DOWN by maximizing NEI class
            targets = torch.LongTensor([correct_label] * len(positive_logits) +
                                       [output_classes - 1] * len(negative_logits)).to(allclass_log_probs.device)
            log_probs_for_loss = torch.gather(allclass_log_probs, -1, targets.unsqueeze(-1))

            if "xtimesneg" in self.config and len(positive_logits) != len(log_probs_for_loss):
                # AVG POSITIVES AND NEGATIVES SEPARATELY
                sent_loss = -(log_probs_for_loss[:len(positive_logits)].mean() + log_probs_for_loss[
                                                                                 len(positive_logits):].mean()) / 2.
            elif self.config.get("sentence_relevance_negative_ss", "") == "khattab_like" and not validation:
                persent_norm_sent_loss = - log_probs_for_loss.mean()

                # shape: # of pos sents x 1 x 3
                # positive_logits
                positive_logits = positive_logits.unsqueeze(1)
                # shape: # of pos sents x # of neg sents x 3
                expanded_neg_logits = negative_logits.unsqueeze(0).expand(
                    (positive_logits.shape[0],) + negative_logits.shape)

                # shape:  # of pos sents x # of neg sents+1 x 3
                single_pos_all_neg = torch.cat((positive_logits, expanded_neg_logits), 1)

                lin_single_pos_all_neg = single_pos_all_neg.view(len(positive_logits), -1)
                renormalized_across_singlepos_allneg_shapedback = F.log_softmax(lin_single_pos_all_neg, 1).view_as(
                    single_pos_all_neg)
                # positive probabilities are in column 0
                perposnegsent_norm_sent_loss = -renormalized_across_singlepos_allneg_shapedback[:, 0,
                                                correct_label].mean(0)
                sent_loss = 0.5 * persent_norm_sent_loss + 0.5 * perposnegsent_norm_sent_loss
            else:
                sent_loss = - log_probs_for_loss.mean()
        else:
            """
            Sum first two rows, consider them "relevant" log_probs. Does not assume that final veracity class is 
            the same as evidence stances.
            """
            # shape of N_of_positive_sentences+N_of_negative_sentences x 2
            relevance_logprobs = torch.stack(
                (torch.logsumexp(allclass_log_probs[:, :2], dim=-1), allclass_log_probs[:, 2]),
                dim=-1)
            # sentence_relevance_logprobs = torch.stack(
            #     (torch.logsumexp(sentence_log_probs[:, :2], dim=-1), sentence_log_probs[:, 2]),
            #     dim=-1)
            # push positives UP by maximizing first dimension and negatives DOWN by maximizing second dimension
            targets = torch.LongTensor([0] * len(positive_logits) + [1] * len(negative_logits)).to(
                allclass_log_probs.device)
            sent_loss = - torch.gather(relevance_logprobs, -1, targets.unsqueeze(-1)).mean()

        losses = {"sent_loss": sent_loss}
        if compute_wl_loss:
            losses["wl_loss"] = wl_loss
        if L_reg_loss is not None:
            losses["lreg_loss"] = L_reg_loss
        if L_sent_reg_loss is not None:
            losses["lreg_sentloss"] = L_sent_reg_loss
        return losses

    # def get_logprobs_per_sentence_from_logits_(self, all_passage_tokens_logits, fid_shaped_src, passage_special_token,
    #                                            sentence_special_token):
    #     """
    #     This contains two for cycles...
    #     """
    #     token_logits_per_sentence = []
    #     sentence_id = 0
    #     start_token = -1
    #     for i in range(len(fid_shaped_src)):
    #         if fid_shaped_src[i] == passage_special_token:
    #             start_token = i
    #         elif fid_shaped_src[i] == sentence_special_token:
    #             token_logits_per_sentence.append(all_passage_tokens_logits[start_token + 1:i])
    #             sentence_id += 1
    #             start_token = i
    #     # sentences x logits
    #     # token_logits_per_sentence is List of sentence representations, each  of size #tokens x 3
    #     """
    #     Do local normalization, and sum to get log-probs for each sentence
    #     """
    #     sentence_logits = torch.stack([
    #         torch.logsumexp(
    #             F.log_softmax(token_logits_in_sentence.view(-1), 0).view(token_logits_in_sentence.shape), 0)
    #         for token_logits_in_sentence in token_logits_per_sentence])
    #     return sentence_logits

    # def extract_unnormalized_logits_(self, all_passage_tokens_3class_logits, fid_shaped_src, passage_special_token,
    #                                  sentence_special_token):
    #     token_logits_per_sentence = []
    #     sentence_id = 0
    #     start_token = -1
    #     for i in range(len(fid_shaped_src)):
    #         if fid_shaped_src[i] == passage_special_token:
    #             start_token = i
    #         elif fid_shaped_src[i] == sentence_special_token:
    #             token_logits_per_sentence.append(all_passage_tokens_3class_logits[start_token + 1:i])
    #             sentence_id += 1
    #             start_token = i
    #     unnormalized_logits = torch.cat(token_logits_per_sentence, dim=0)
    #     return unnormalized_logits

    def extract_unnormalized_logits(self, all_passage_tokens_class_logits, fid_shaped_src, passage_special_token,
                                    sentence_special_token):
        token_logits_per_sentence_idx = []
        start_token = -1
        for i in range(len(fid_shaped_src)):
            if fid_shaped_src[i] == passage_special_token:
                start_token = i
            elif fid_shaped_src[i] == sentence_special_token:
                token_logits_per_sentence_idx.extend(list(range(start_token + 1, i)))
                start_token = i
        unnormalized_logits = all_passage_tokens_class_logits[token_logits_per_sentence_idx]
        return unnormalized_logits

    @torch.no_grad()
    def get_evidence_mask(self, src, metadata, mask_nongts=False):
        passage_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.passage_special_token)
        sentence_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sentence_special_token)

        if mask_nongts and metadata is not None:
            if type(metadata) == list:
                assert len(metadata) == 1
                metadata = metadata[0]
            gt_indices = []
            each_passage_sentence_count = [len(x.split("|")) for x in metadata['sent_ranges']]
            for passage_idx, sent_idx in metadata['relevant_sentence_labels']:
                flat_sentence_index = sum(each_passage_sentence_count[:passage_idx]) + sent_idx
                gt_indices.append(flat_sentence_index)

            evidence_mask = []
            sentence_index = 0
            for inp in src:
                mask_per_inp = []
                passage_found = False
                sentence_index -= 1
                for token in inp:
                    mask_per_inp.append(
                        passage_found and token != sentence_special_token and sentence_index in gt_indices and
                        token != self.tokenizer.pad_token_id and token != self.tokenizer.sep_token_id)
                    if token == sentence_special_token:
                        sentence_index += 1
                    if token == passage_special_token:
                        passage_found = True
                        sentence_index += 1
                evidence_mask.append(mask_per_inp)
            return torch.BoolTensor(evidence_mask).to(src.device)
        else:
            evidence_mask = []
            for inp in src:
                mask_per_inp = []
                passage_found = False
                for token in inp:
                    mask_per_inp.append(
                        passage_found and token != sentence_special_token and
                        token != self.tokenizer.pad_token_id and token != self.tokenizer.sep_token_id)
                    if token == passage_special_token:
                        passage_found = True
                evidence_mask.append(mask_per_inp)
            return torch.BoolTensor(evidence_mask).to(src.device)

    def get_logprobs_per_sentence_from_logits(self, all_passage_tokens_logits, fid_shaped_src, passage_special_token,
                                              sentence_special_token, considered_indices=None,
                                              return_perword_logprobs=False,
                                              paragraph_supervision=False, hard_paragraph_supervision=False,
                                              true_hardem=False, stochastic_em=False, return_perword_logits=False):
        """
        This implementation uses indexing, no for loops
        Benchmarking:
        # For loop implementation
        # list: 50x: 1:02
        # tensor: 50x: 1:11

        # Index implementation with list as fid_shaped_src
        # 50x:  1:02
        """
        # mask out CLS
        all_passage_tokens_logits[0] = -float("inf")
        output_classes = 3

        if paragraph_supervision:
            token_logits_per_provenance_idx = []
            token_logits_per_single_provenance_idx = []
            if hard_paragraph_supervision:
                token_logits_per_provenances_idx = []
            start_token = -1
            for i in range(len(fid_shaped_src)):
                if fid_shaped_src[i] == passage_special_token:
                    start_token = i
                    if token_logits_per_single_provenance_idx:
                        # Append all tokens from all sentences from this block and start again
                        token_logits_per_provenance_idx.append(flatten(token_logits_per_single_provenance_idx))
                        if hard_paragraph_supervision:
                            token_logits_per_provenances_idx.append(token_logits_per_single_provenance_idx)
                        token_logits_per_single_provenance_idx = []
                elif fid_shaped_src[i] == sentence_special_token:
                    # Collect per-sentence tokens
                    token_logits_per_single_provenance_idx.append(list(range(start_token + 1, i)))
                    start_token = i
            # add last paragraph
            assert token_logits_per_single_provenance_idx
            token_logits_per_provenance_idx.append(flatten(token_logits_per_single_provenance_idx))
            if hard_paragraph_supervision:
                token_logits_per_provenances_idx.append(token_logits_per_single_provenance_idx)
            token_logits_per_single_provenance_idx = []

            if hard_paragraph_supervision:
                """
                Keep track of which sentence belongs to which block
                """
                token_logits_per_provenance_idx_ = []
                sentence_provenance_indices = []
                for provenance_idx, token_logits_per__provenance_idx in enumerate(token_logits_per_provenances_idx):
                    for token_logits_per_sentence_idx in token_logits_per__provenance_idx:
                        token_logits_per_provenance_idx_.append(token_logits_per_sentence_idx)
                        sentence_provenance_indices.append(provenance_idx)
                # Convert indices to ranges (from which to which sentence there is a block i)
                sentences_per_paragraph_ranges = []
                _from = 0
                _last_idx = -1
                for idx, prov_idx in enumerate(sentence_provenance_indices):
                    if idx > 0 and prov_idx == _last_idx + 1:
                        sentences_per_paragraph_ranges.append([_from, idx])
                        _from = idx
                    _last_idx = prov_idx
                sentences_per_paragraph_ranges.append([_from, len(sentence_provenance_indices)])

                token_logits_per_provenance_idx = token_logits_per_provenance_idx_
        else:
            token_logits_per_provenance_idx = []
            start_token = -1
            for i in range(len(fid_shaped_src)):
                if fid_shaped_src[i] == passage_special_token:
                    start_token = i
                elif fid_shaped_src[i] == sentence_special_token:
                    token_logits_per_provenance_idx.append(list(range(start_token + 1, i)))
                    start_token = i

        total_provenances = len(token_logits_per_provenance_idx)
        longest_provenance = max(len(x) for x in token_logits_per_provenance_idx)

        padded_linearized_token_logits_perprovenance_idx = flatten(
            x + [0] * (longest_provenance - len(x)) for x in token_logits_per_provenance_idx)
        token_logits_perprovenance = all_passage_tokens_logits[padded_linearized_token_logits_perprovenance_idx].view(
            total_provenances, longest_provenance, output_classes).float()

        """ 
        Do local normalization, and sum to get log-probs for each sentence
        """

        if self.config.get("paper_baseline", False):
            """
            In training, we only consider annotated provenances, passed via "considered_indices" variable
            """
            if considered_indices is not None:
                token_logits_perprovenance = token_logits_perprovenance[considered_indices]
            token_logits_perprovenance_lin = token_logits_perprovenance.view(-1)
            provenance_logprobs_per_word = F.log_softmax(token_logits_perprovenance_lin, 0).view(
                token_logits_perprovenance.shape)
        else:
            token_logits_perprovenance_lin = token_logits_perprovenance.view(total_provenances, -1)
            if hard_paragraph_supervision and true_hardem:
                """
                Logsoftmax over whole blocks, not just sentences
                """
                assert paragraph_supervision
                provenance_logprobs_per_word = torch.cat([F.log_softmax(token_logits_perprovenance_lin[s:e].view(-1), 0)
                                                         .view(token_logits_perprovenance_lin[s:e].shape)
                                                          for s, e in sentences_per_paragraph_ranges], 0).view(
                    token_logits_perprovenance.shape)
            else:
                provenance_logprobs_per_word = F.log_softmax(token_logits_perprovenance_lin, 1).view(
                    token_logits_perprovenance.shape)
        if self.config.get("perword_maxpool", False):
            provenance_logits = provenance_logprobs_per_word.max(dim=1).values
        elif self.config.get("perword_topkpool", False): \
                provenance_logits = torch.logsumexp(provenance_logprobs_per_word.topk(k=5, dim=1).values, 1)
        else:  # logsumexp
            provenance_logits = torch.logsumexp(provenance_logprobs_per_word, 1)
            if hard_paragraph_supervision:
                with torch.no_grad():
                    relevance_logprobs = torch.logsumexp(provenance_logits[:, :2], -1)
                    if stochastic_em:  # sample supervision according to sentence probs
                        selected_indices = [get_samples_cumsum(relevance_logprobs[s:e].exp().cpu().numpy()).item() + s
                                            for s, e in
                                            sentences_per_paragraph_ranges]
                    else:  # select most probable sentences in block as supervision
                        selected_indices = [relevance_logprobs[s:e].argmax().item() + s for s, e in
                                            sentences_per_paragraph_ranges]
                provenance_logits = provenance_logits[selected_indices]
        provenance_logits = (provenance_logits,)
        if return_perword_logprobs:
            provenance_logits = provenance_logits + (provenance_logprobs_per_word,)
        if return_perword_logits:
            provenance_logits = provenance_logits + (token_logits_perprovenance,)
        return provenance_logits

    def compute_wordlevel_loss(self, perword_logprobs, annotated, fid_shaped_src_list, gt_indices, labels):
        # passage_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.passage_special_token)
        # sentence_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sentence_special_token)
        # tokens_per_sentence = []
        # sequences = []
        # start_token = -1
        # for i in range(len(fid_shaped_src_list)):
        #     if fid_shaped_src_list[i] == passage_special_token:
        #         start_token = i
        #     elif fid_shaped_src_list[i] == sentence_special_token:
        #         tokens_per_sentence.append(
        #             [self.tokenizer._convert_id_to_token(i) for i in fid_shaped_src_list[start_token + 1:i]])
        #         sequences.append(fid_shaped_src_list[start_token + 1:i])
        #         start_token = i
        #
        # print(f"C: {annotated['claim']}")
        # for k, v in annotated['label'].items():
        #     key = gt_indices[int(k)]
        #     print(" ".join(tokens_per_sentence[key]))
        #     print(" ".join([x for m, x in zip(v, tokens_per_sentence[key]) if m]))
        loss_wl = 0

        assert labels[0] in [0, 1]
        normalizer = 0
        for k, v in annotated['label'].items():
            ann_idx = gt_indices[int(k)]
            # Multiply probs for relevant words (= sum logprobs)
            if not len(v) <= len(perword_logprobs[ann_idx]):
                # sometimes tokenization-detokenization errors can happen
                # i.e. self.tokenizer.encode(self.tokenizer.decode(t)) != t
                # workaround them for now, but this should be fixed later, if this experiment will work!
                # TODO: Martin
                # UPDATE, doesnt seem that supervision is helping at all
                v = v[:len(perword_logprobs[ann_idx])]
            annotated_words_in_sentence_logprobs = perword_logprobs[ann_idx][:len(v)][v][:, labels[0]]
            loss_per_sentence = annotated_words_in_sentence_logprobs.sum()
            if not torch.isinf(loss_per_sentence):  # tokenization-detokenization errors
                loss_wl += loss_per_sentence
                normalizer += len(annotated_words_in_sentence_logprobs)
        if normalizer < 1e-6:  # avoid division by zero
            normalizer = 1.
        return -loss_wl / normalizer

    @torch.no_grad()
    def validate(self, *args, **kwargs):
        if self.config["dataset"] == "fever":
            return self.validate_FEVER(*args, **kwargs)
        elif self.config['dataset'] == 'hover':
            return self.validate_HOVER(*args, **kwargs)
        elif self.config["dataset"] == "faviq":
            return self.validate_FAVIQ(*args, **kwargs)
        elif self.config["dataset"] == "realfc":
            return self.validate_REALFC(*args, **kwargs)

        else:
            raise ValueError("Unknown dataset!")

    # @profile
    @torch.no_grad()
    def validate_REALFC(self, model: TransformerClassifier, val_iter, optimizer_dict=None,
                        log_results=False):
        improvement = False
        model = model.eval()
        for param in model.parameters():
            param.grad = None

        official_ids = []
        gts, dataset_predictions = [], []
        conflicting_evidence_mask = []
        losslist = []

        passage_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.passage_special_token)
        title_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.title_special_token)

        gt_dict = {"supported": 0, "refuted": 1, "neutral": 2}
        relclass = {"supporting": 0, "refuting": 1, "neutral": 2}

        if log_results:
            import xlsxwriter
            from colour import Color
            mkdir(".results")
            fn = f".results/realfc_lrm_verifier_{log_results}"
            if self.distributed:
                fn += f"_{self.global_rank}"

            logs = []

        it = tqdm(enumerate(val_iter), total=ceil(len(val_iter.dataset) / val_iter.batch_size))
        for i, (src, src_type_ids, src_mask, label, metadata) in it:
            official_ids.append((metadata[0]["ncid"], metadata[0]["section_name"]))
            if self.config.get("log_total_results", False):
                if i > self.config["log_total_results"]:
                    break
            all_sentences_neutral = all(l == 'neutral' for l in metadata[0]['sentence_labels'])
            if self.config.get("skip_allneutral_sections", False) and all_sentences_neutral:
                continue
            is_conflictingevidence = "supporting" in metadata[0]["sentence_labels"] and \
                                     "refuting" in metadata[0]["sentence_labels"]
            if self.config.get("skip_nonconflicting_evidence", False) and not is_conflictingevidence:
                continue

            # Move to gpu
            src, src_mask = src.to(self.torch_device), src_mask.to(
                self.torch_device)
            if src_type_ids is not None:
                src_type_ids = src_type_ids.to(self.torch_device)
            if self.config.get("disable_token_type_ids", False):
                src_type_ids = []
            conflicting_evidence_mask.append(is_conflictingevidence)

            src = src[0]
            src_mask = src_mask[0]
            if src_type_ids is not None:
                src_type_ids = src_type_ids[0]

            loss, validation_outputs = self.forward_pass(src, src_type_ids, src_mask, label, metadata, model,
                                                         validation=True)
            losslist.append(loss.item())

            # Get predictions
            sentence_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sentence_special_token)
            passage_mask = src_mask.bool().view(-1)
            fid_shaped_src = src.view(-1)[passage_mask]
            fid_shaped_src_list = fid_shaped_src.tolist()
            class_logits, class_labels = validation_outputs['marg_logprobs'], validation_outputs['marg_labels']
            output_logits = validation_outputs['output_logits']
            """
            Forward output already contains sentence-level normalized log-probs
            """
            sentence_logprobs, sentence_perword_logprobs, sentence_perword_logits = self.get_logprobs_per_sentence_from_logits(
                output_logits.squeeze(0),
                fid_shaped_src_list,
                passage_special_token,
                sentence_special_token,
                return_perword_logprobs=True,
                return_perword_logits=True)

            linear_combinations_per_provenance = \
                (sentence_perword_logits - sentence_perword_logprobs)[:, 0, 0].exp()
            relevant_sentence_indices = torch.argmax(sentence_logprobs, -1).tolist()
            sentence_logprobs_binary = torch.stack(
                (torch.logsumexp(sentence_logprobs[:, :2], dim=-1), sentence_logprobs[:, 2]), dim=-1)
            relevant_sentence_indices_binary = (1 - torch.argmax(sentence_logprobs_binary, -1)).tolist()

            pred_numerical_category = torch.argmax(class_logits).item()
            gt_numerical_category = gt_dict[label[0]]

            gt = (gt_numerical_category, [relclass[l] for l in metadata[0]['sentence_labels']])
            gts.append(gt)
            dataset_prediction = (pred_numerical_category, relevant_sentence_indices, relevant_sentence_indices_binary)
            dataset_predictions.append(dataset_prediction)

            # Write logs
            if log_results:
                input_texts, sentences_in_input_texts, titles_in_input_texts = self.get_input_text_sentences(
                    fid_shaped_src, passage_special_token, sentence_special_token, title_special_token)
                # Assume global log_softmax + per-sentence softmax
                with torch.cuda.amp.autocast():
                    input_sentence_scores = sentence_logprobs.exp()
                claim = self.tokenizer.decode(src[0][1:src[0].cpu().tolist().index(title_special_token)])

                offset_encodings = [self.tokenizer.encode_plus(" " + t, return_offsets_mapping=True,
                                                               add_special_tokens=False).offset_mapping for t in
                                    input_texts]
                log = {
                    "correct": pred_numerical_category == gt_numerical_category,
                    "evidence_correct": all(x == y for x, y in zip(relevant_sentence_indices, gt[1])),
                    "claim": claim,
                    "label": gt_numerical_category,
                    "predicted_label": pred_numerical_category,
                    "input_sentences": (input_texts, sentences_in_input_texts, titles_in_input_texts),
                    "input_sentence_scores": input_sentence_scores,
                    "linear_combinations_per_provenance": linear_combinations_per_provenance,
                    "metadata": metadata[0]
                }
                log['input_sentences_perword_probs'] = sentence_perword_logprobs.exp()
                log['input_text_offset_encodings'] = offset_encodings
                logs.append(log)
        if log_results:
            self.write_logs(fn, logs)
            del logs

        if self.distributed:
            dist_losslist = cat_lists(
                share_list(losslist, rank=self.global_rank, world_size=self.world_size))
            dist_gts = cat_lists(
                share_list(gts, rank=self.global_rank, world_size=self.world_size))
            dist_preds = cat_lists(
                share_list(dataset_predictions, rank=self.global_rank, world_size=self.world_size))
            dist_officids = cat_lists(
                share_list(official_ids, rank=self.global_rank, world_size=self.world_size))
            dist_conflicting_evidence_mask = cat_lists(
                share_list(conflicting_evidence_mask, rank=self.global_rank, world_size=self.world_size))
            official_ids = dist_officids
            losslist = dist_losslist
            gts = dist_gts
            dataset_predictions = dist_preds
            conflicting_evidence_mask = dist_conflicting_evidence_mask

        # Compute final evaluation
        EviF1, VAcc, VF1, accuracy, c_EviF1, c_VAcc, c_VF1, c_condACC, c_condF1, condACC, condF1 = self.report_results(
            dataset_predictions, gts, losslist, official_ids)
        logger.info("Total conflicting evidences: %d" % sum(conflicting_evidence_mask))
        self.report_results([x for x, m in zip(dataset_predictions, conflicting_evidence_mask) if m],
                            [x for x, m in zip(gts, conflicting_evidence_mask) if m],
                            [x for x, m in zip(losslist, conflicting_evidence_mask) if m],
                            [x for x, m in zip(official_ids, conflicting_evidence_mask) if m],
                            prefix="ConflictingEvi: ")
        # logger.debug(json.dumps([x for x, m in zip(official_ids, conflicting_evidence_mask) if m], indent=4))

        if not "score_focus" in self.config:
            best_score = max(c_condF1, condF1)
        else:
            scores = {
                "condF1": max(c_condF1, condF1),
                "condACC": max(c_condACC, condACC),
                "VF1": max(c_VF1, VF1),
                "VAcc": max(c_VAcc, VAcc),
                "EviF1": max(c_EviF1, EviF1)
            }
            best_score = scores[self.config["score_focus"]]

        if not self.config['test_only']:
            if self.config.get("save_all_checkpoints", False) and (not self.distributed or self.global_rank == 0):
                self.save_model(accuracy, model, optimizer_dict, "", False, best_score)

            if best_score > self.best_score:
                logger.info(f"{best_score} ---> New BEST!")
                self.best_score = best_score
                improvement = True
                # Saving disabled for now
                if (not self.distributed or self.global_rank == 0) and not self.config.get("save_all_checkpoints",
                                                                                           False):
                    self.save_model(accuracy, model, optimizer_dict, "", False, best_score)
        if self.distributed:
            # do not start training with other processes
            # this is useful when process 0 saves the model,
            # if there is no barrier other processes already do training
            dist.barrier()
        model = model.train()
        return best_score, improvement

    def report_results(self, dataset_predictions, gts, losslist, official_ids, prefix=""):
        loss = sum(losslist) / len(losslist)
        accuracy = sum(pred[0] == gt[0] for pred, gt in zip(dataset_predictions, gts)) / len(dataset_predictions)
        logger.info(f"{prefix}RealFC Evaluation:\n"
                    f"Loss: {loss:.3f} ACC: {accuracy:.3f}\n")
        report_corrected, report_corrected_json, report_normal, report_normal_json = \
            eval_realfc_official(dataset_predictions, gts, official_ids)
        if self.config.get("test_only", False):
            logger.debug("----DETAILED VALIDATION REPORT-----")
            logger.debug("----------REPORT NORMAL------------")
            logger.debug(report_normal)
            logger.debug("----------REPORT CORRECTED---------")
            logger.debug(report_corrected)
            # Log out json metrics
            logger.debug(json.dumps({
                'corrected': report_corrected_json,
                'uncorrected': report_normal_json
            }, indent=4))

        def extract_report_metrics(r):
            return r['weighted_f1']["avg-weighted-f1"], \
                r['weighted_f1']["weighted_accuracy"], \
                r['metrics-isolated-veracity']["macro avg"]["f1-score"], \
                r['metrics-isolated-veracity']["accuracy"], \
                r['metrics-isolated-evidence']["evidence"]["f1-score"]

        condF1, condACC, VF1, VAcc, EviF1 = extract_report_metrics(report_normal_json)
        c_condF1, c_condACC, c_VF1, c_VAcc, c_EviF1 = extract_report_metrics(report_corrected_json)
        if not self.distributed or self.global_rank == 0:
            logger.info(f"{prefix}Uncorrected Metrics:\n"
                        f"{prefix}condF1: {condF1:.5f}\n"
                        f"{prefix}condACC: {condACC:.5f}\n"
                        f"{prefix}VF1: {VF1:.5f}\n"
                        f"{prefix}VAcc: {VAcc:.5f}\n"
                        f"{prefix}EviF1: {EviF1:.5f}\n")
            logger.info(f"{prefix}Corrected Metrics:\n"
                        f"{prefix}condF1: {c_condF1:.5f}\n"
                        f"{prefix}condACC: {c_condACC:.5f}\n"
                        f"{prefix}VF1: {c_VF1:.5f}\n"
                        f"{prefix}VAcc: {c_VAcc:.5f}\n"
                        f"{prefix}EviF1: {c_EviF1:.5f}\n")
        return EviF1, VAcc, VF1, accuracy, c_EviF1, c_VAcc, c_VF1, c_condACC, c_condF1, condACC, condF1

    @torch.no_grad()
    def validate_FEVER(self, model: TransformerClassifier, val_iter, optimizer_dict=None,
                       log_results=False):
        improvement = False
        model = model.eval()
        for param in model.parameters():
            param.grad = None

        if self.config.get("train_masker", False):
            best_score = self.validate_interpretability_FEVER(model, val_iter)
            if self.config.get("save_all_checkpoints", False) and (not self.distributed or self.global_rank == 0):
                self.save_model(best_score, model, optimizer_dict, 0, 0, 0)

            if best_score > self.best_score:
                improvement = True
                logger.info(f"{best_score} ---> New BEST!")
                self.best_score = best_score
                # Saving disabled for now
                if (not self.distributed or self.global_rank == 0) and not self.config.get("save_all_checkpoints",
                                                                                           False):
                    self.save_model(best_score, model, optimizer_dict, 0, 0, 0)
            return best_score, improvement
        elif self.config.get("eval_interp_during_training", False):
            val_iter, val_iter_interp = val_iter
            interp_score = self.validate_interpretability_FEVER(model, val_iter_interp)
            logger.info(f"INTERP-SCORE: {interp_score:.2f}")

        recallat5_hits = 0  # not considering NEI
        paragraph_recall5_hits = 0
        recallat5_samples = 0
        multihop_mask = []

        gts, preds = [], []
        dataset_predictions = []

        reranking_only = self.config.get("reranking_only", False)
        online_r5 = 0.

        losslist = []
        passage_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.passage_special_token)
        title_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.title_special_token)
        if log_results:
            import xlsxwriter
            from colour import Color
            mkdir(".results")
            fn = f".results/lrm_verifier_{log_results}"
            if self.distributed:
                fn += f"_{self.global_rank}"

            logs = []
        if self.config.get("log_predictions_conflicting_evidence", False):
            fn = ".results/lrm_verifier_conflictingevicence.csv"
            log_fhandle = open(fn, "w")
            csvwriter = csv.writer(log_fhandle, delimiter=",")
        if self.config.get("dump_topk_relevances_predictionscores", False):
            relevance_scores = []
            prediction_scores = []
        if "dump_topk_notcorrelated_sentences" in self.config:
            with open(self.config["dump_topk_notcorrelated_sentences"], "rb") as f:
                notcorrelated_ex_indices = set(pickle.load(f))
            fn = "top200samples_notcorrelated_sentences.csv"
            log_fhandle = open(fn, "w")
            csvwriter = csv.writer(log_fhandle, delimiter=",")
        gt_dict = gt_dicts[self.config["dataset"]]

        it = tqdm(enumerate(val_iter), total=ceil(len(val_iter.dataset) / val_iter.batch_size))

        for i, (src, src_type_ids, src_mask, label, metadata) in it:
            if self.config.get("log_total_results", False):
                if i > self.config["log_total_results"]:
                    break
            if "dump_topk_notcorrelated_sentences" in self.config:
                if not i in notcorrelated_ex_indices:
                    continue
            # Move to gpu
            src, src_mask = src.to(self.torch_device), src_mask.to(
                self.torch_device)

            if src_type_ids is not None:
                src_type_ids = src_type_ids.to(self.torch_device)

            if self.config.get("disable_token_type_ids", False):
                src_type_ids = []
            if self.config['dataset'] == "fever":
                is_multihop = label[0] != 'NOT ENOUGH INFO' and all(len(e) > 1 for e in metadata[0]['evidence'])
                is_multihop_article_sample = all(len(set(ex[-2] for ex in e)) > 1 for e in metadata[0]['evidence'])
                multihop_mask.append(is_multihop)
            assert self.config['dataset'] != "hover"

            src = src[0]
            src_mask = src_mask[0]
            if src_type_ids is not None:
                src_type_ids = src_type_ids[0]

            loss, validation_outputs = self.forward_pass(src, src_type_ids, src_mask, label, metadata, model,
                                                         validation=True)

            losslist.append(loss.item())

            #######
            # Get predicted evidence
            sentence_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sentence_special_token)
            passage_mask = src_mask.bool().view(-1)
            fid_shaped_src = src.view(-1)[passage_mask]
            fid_shaped_src_list = fid_shaped_src.tolist()
            sentences_at_input = (src == sentence_special_token).sum(-1).cumsum(0)
            sentence_tokens_mask = (fid_shaped_src == sentence_special_token)

            if self.config.get("dump_attention_matrices", False):
                attention_matrix = validation_outputs["attention_weights"]
                self.dump_attention_matrix(attention_matrix, src, fid_shaped_src, sentences_at_input, metadata)

            class_logits, class_labels = validation_outputs['marg_logprobs'], validation_outputs['marg_labels']
            output_logits = validation_outputs['output_logits']
            """
            Forward output already contains sentence-level normalized log-probs
            """
            sentence_logprobs, sentence_perword_logprobs, sentence_perword_logits = self.get_logprobs_per_sentence_from_logits(
                output_logits.squeeze(0),
                fid_shaped_src_list,
                passage_special_token,
                sentence_special_token,
                return_perword_logprobs=True,
                return_perword_logits=True)

            linear_combinations_per_provenance = \
                (sentence_perword_logits - sentence_perword_logprobs)[:, 0, 0].exp()
            orig_sentence_logprobs = sentence_logprobs
            if self.config.get("paragraph_supervision", False):
                paragraph_logprobs, paragraph_perword_logprobs = self.get_logprobs_per_sentence_from_logits(
                    output_logits.squeeze(0),
                    fid_shaped_src_list,
                    passage_special_token,
                    sentence_special_token,
                    paragraph_supervision=True,
                    return_perword_logprobs=True)

            if not self.config.get("predict_evidence_according_class", False):
                sentence_logprobs = torch.stack(
                    (torch.logsumexp(sentence_logprobs[:, :2], dim=-1), sentence_logprobs[:, 2]), dim=-1)
                if self.config.get("paragraph_supervision", False):
                    paragraph_logprobs = torch.stack(
                        (torch.logsumexp(paragraph_logprobs[:, :2], dim=-1), paragraph_logprobs[:, 2]), dim=-1)

            if self.config.get("predict_top5_sentences", False):
                if self.config.get("predict_evidence_from_logits", False):
                    """
                    Predict according to logits, not according to sentence-normalized probabilities
                    Logits are logs of per-sentence probabilities scaled by linear coefficient about final prediction
                    """
                    logits_per_provenance = torch.logsumexp(sentence_perword_logits, 1)
                    if not self.config.get("predict_evidence_according_class", False):
                        relevant_sentence_scores, relevant_sentence_indices = torch.topk(
                            torch.logsumexp(logits_per_provenance[:, :2], dim=-1),
                            k=5 * 10)
                    else:
                        predicted_SR_class = torch.argmax(class_logits[:2]).item()
                        relevant_sentence_scores, relevant_sentence_indices = torch.topk(
                            logits_per_provenance[:, predicted_SR_class], k=5 * 10)
                elif self.config.get("predict_evidence_according_class", False):
                    predicted_SR_class = torch.argmax(class_logits[:2]).item()

                    relevant_sentence_scores, relevant_sentence_indices = torch.topk(
                        sentence_logprobs[:, predicted_SR_class], k=5 * 10)
                elif self.config.get("predict_evidence_from_combination", False):
                    relevant_sentence_scores, relevant_sentence_indices = torch.topk(
                        linear_combinations_per_provenance, k=5 * 10)
                else:
                    relevant_sentence_scores, relevant_sentence_indices = torch.topk(sentence_logprobs[:, 0],
                                                                                     k=5 * 10)
                    if self.config.get("paragraph_supervision", False):
                        relevant_paragraph_scores, relevant_paragraph_indices = paragraph_logprobs[:, 0].sort(
                            descending=True)
                    # if self.config.get("log_predictions_conflicting_evidence", False):
                    #     relevant_supporting_scores, relevant_supporting_indices = torch.topk(
                    #         orig_sentence_logprobs[:, gt_dict["SUPPORTS"]],
                    #         k=5 * 10)
                    #     relevant_refuting_scores, relevant_refuting_indices = torch.topk(
                    #         orig_sentence_logprobs[:, gt_dict["REFUTES"]],
                    #         k=5 * 10)
            else:
                relevant_sentence_indices = ~ torch.argmax(sentence_logprobs, -1).bool()

            if self.config.get("dump_topk_relevances_predictionscores", False):
                threshold = self.config["topk_threshold"]
                if class_labels[0] == gt_dict["NOT ENOUGH INFO"]:
                    relevance_scores.append([])
                    prediction_scores.append([])
                else:
                    relevant_sentence_probs = relevant_sentence_scores.exp()
                    relevance_scores.append([s for s in relevant_sentence_probs.tolist() if s > threshold])
                    relevant_sentence_scores_above_threshold = [s.item() > threshold for s in relevant_sentence_probs]
                    indices_of_rss_above_threshold = [x for x, above in zip(relevant_sentence_indices.tolist(),
                                                                            relevant_sentence_scores_above_threshold) if
                                                      above]
                    prediction_scores.append(
                        [linear_combinations_per_provenance[i].item() for i in indices_of_rss_above_threshold])
            elif "dump_topk_notcorrelated_sentences" in self.config:
                threshold = self.config["topk_threshold"]
                claim = self.tokenizer.decode(src[0][1:src[0].cpu().tolist().index(title_special_token)])
                assert class_labels[0] != gt_dict["NOT ENOUGH INFO"]
                if class_labels[0] != gt_dict["NOT ENOUGH INFO"]:
                    relevant_sentence_probs = relevant_sentence_scores.exp()
                    relevant_sentence_probs_filtered = [s for s in relevant_sentence_probs.tolist() if s > threshold]
                    relevant_sentence_scores_above_threshold = [s.item() > threshold for s in relevant_sentence_probs]
                    indices_of_rss_above_threshold = [x for x, above in zip(relevant_sentence_indices.tolist(),
                                                                            relevant_sentence_scores_above_threshold) if
                                                      above]
                    prediction_scores_for_sentences = np.array([linear_combinations_per_provenance[i].item() for i in
                                                                indices_of_rss_above_threshold])
                    ps_ranked_indices = np.argsort(prediction_scores_for_sentences)[::-1]  # descending
                    ps_ranked_scores = prediction_scores_for_sentences[ps_ranked_indices].tolist()
                    indices_of_ps_above_threshold = [indices_of_rss_above_threshold[i] for i in ps_ranked_indices]

                    input_texts, sentences_in_input_texts, titles_in_input_texts = self.get_input_text_sentences(
                        fid_shaped_src, passage_special_token, sentence_special_token, title_special_token)
                    input_texts = [input_texts[i] for i in sentences_in_input_texts]

                    gt_indices = []
                    each_passage_sentence_count = [len(x.split("|")) for x in metadata[0]['sent_ranges']]
                    for passage_idx, sent_idx in metadata[0]['relevant_sentence_labels']:
                        flat_sentence_index = sum(each_passage_sentence_count[:passage_idx]) + sent_idx
                        gt_indices.append(flat_sentence_index)
                    gt_texts = [input_texts[i] for i in gt_indices]

                    input_texts_rs = [input_texts[i] for i in indices_of_rss_above_threshold]
                    titles_rs = [titles_in_input_texts[i] for i in indices_of_rss_above_threshold]
                    sentences_rs = "\n".join(
                        [f"({rank}) {t}|{s}" for rank, (s, t) in enumerate(zip(input_texts_rs, titles_rs))])

                    input_texts_ps = [input_texts[i] for i in indices_of_ps_above_threshold]
                    titles_ps = [titles_in_input_texts[i] for i in indices_of_ps_above_threshold]
                    sentences_ps = "\n".join(
                        [f"({rank}) {t}|{s}" for rank, (s, t) in enumerate(zip(input_texts_ps, titles_ps))])

                    csvwriter.writerow(
                        [claim, sentences_rs, relevant_sentence_probs_filtered, sentences_ps, ps_ranked_scores,
                         "\n".join(gt_texts), class_labels[0]])
            if not self.config.get("predict_top5_sentences", False):
                relevant_sentence_indices = [si for si, s in enumerate(relevant_sentence_indices) if s]
            else:
                if self.config.get("paragraph_supervision", False):
                    if class_labels[0] != gt_dict["NOT ENOUGH INFO"]:
                        relevant_paragraph_indices = relevant_paragraph_indices.tolist()

                        predicted_paragraph_titles = [metadata[0]['titles'][i] for i in relevant_paragraph_indices][:5]
                        gt_title_annotations = [list({title for _, _, title, _ in annotation}) for annotation in
                                                metadata[0]['evidence']]
                        paragraph_hit = any(
                            all(gttitle in predicted_paragraph_titles for gttitle in gt_titles) for gt_titles in
                            gt_title_annotations)
                        paragraph_recall5_hits += int(paragraph_hit)
                relevant_sentence_indices = relevant_sentence_indices.tolist()

            sentences_at_input = sentences_at_input.tolist()
            deduplicated_scores, predicted_evidence = self.get_predicted_evidence(metadata,
                                                                                  relevant_sentence_indices,
                                                                                  relevant_sentence_scores,
                                                                                  sentences_at_input)
            # if self.config.get("log_predictions_conflicting_evidence", False):
            #     supporting_scores, supporting_evidence = self.get_predicted_evidence(metadata,
            #                                                                          relevant_supporting_indices,
            #                                                                          relevant_supporting_scores,
            #                                                                          sentences_at_input)
            #     refuting_scores, refuting_evidence = self.get_predicted_evidence(metadata,
            #                                                                      relevant_refuting_indices,
            #                                                                      relevant_refuting_scores,
            #                                                                      sentences_at_input)
            #     K_log = 10
            #     supporting_scores, supporting_evidence = supporting_scores[:K_log], supporting_evidence[:K_log]
            #     refuting_scores, refuting_evidence = refuting_scores[:K_log], refuting_evidence[:K_log]

            predicted_evidence = [list(e) for e in predicted_evidence]
            #### end of  getting predicted evidence
            if self.config.get("predict_top5_sentences", False):
                predicted_evidence = predicted_evidence[:5]
                assert len(predicted_evidence) == 5
                predicted_evidence_scores = deduplicated_scores[:5]
            if self.config.get("do_IDK_toprel_eval", False):
                not_a_NEI_score = torch.logsumexp(torch.stack(predicted_evidence_scores),
                                                  0).item()  # higher the score, the lower the probability that the sample is NEI
                # not_a_NEI_score = torch.sum(torch.stack(predicted_evidence_scores),
                #                             0).item()
                gt_numerical_category = gt_dict[label[0]]

                preds.append((class_logits, not_a_NEI_score))
                gts.append(gt_numerical_category)

                fever_prediction = {
                    "label": label[0],
                    "predicted_label": None,
                    "evidence": metadata[0]['evidence'],
                    "predicted_evidence": predicted_evidence,
                }
                dataset_predictions.append(fever_prediction)
            elif not reranking_only:
                sr_numerical_category = torch.argmax(class_logits).item()
                gt_numerical_category = gt_dict[label[0]]
                # if prediction_numerical_category == gt_dict["NOT ENOUGH INFO"]:
                #     predicted_evidence = []

                gt_list = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
                preds.append(sr_numerical_category)
                gts.append(gt_numerical_category)

                fever_prediction = {
                    "label": label[0],
                    "predicted_label": gt_list[sr_numerical_category],
                    "evidence": metadata[0]['evidence'],
                    "predicted_evidence": predicted_evidence,
                }
                if self.config.get("log_predictions_conflicting_evidence", False):
                    log_ex = {
                        "label": label[0],
                        "predicted_label": gt_list[sr_numerical_category],
                        "evidence": metadata[0]['evidence'],
                        "predicted_evidence": predicted_evidence,
                    }

                dataset_predictions.append(fever_prediction)

            if class_labels[0] != gt_dict["NOT ENOUGH INFO"]:
                def is_evidence_correct(ground_truth_evidence, predicted_evidence, max_evidence=5):
                    for evidence_group in ground_truth_evidence:
                        # Filter out the annotation ids. We just want the evidence page and line number
                        actual_sentences = [[e[2], e[3]] for e in evidence_group]
                        # Only return true if an entire group of actual sentences is in the predicted sentences
                        if all([actual_sent in predicted_evidence[:max_evidence] for actual_sent in
                                actual_sentences]):
                            return True
                    return False

                evidence_correct = is_evidence_correct(metadata[0]['evidence'], predicted_evidence,
                                                       max_evidence=5)
                recallat5_hits += int(evidence_correct)
                recallat5_samples += 1

                assert len(label) == 1

                online_r5 = recallat5_hits / recallat5_samples if recallat5_samples > 0 else 1.
            assert len(label) == 1
            if log_results or self.config.get("log_predictions_conflicting_evidence", False):
                input_texts, sentences_in_input_texts, titles_in_input_texts = self.get_input_text_sentences(
                    fid_shaped_src, passage_special_token, sentence_special_token, title_special_token)
                # Assume global log_softmax + per-sentence softmax
                with torch.cuda.amp.autocast():
                    input_sentence_scores = orig_sentence_logprobs.exp()
                claim = self.tokenizer.decode(src[0][1:src[0].cpu().tolist().index(title_special_token)])
                if log_results:

                    offset_encodings = [self.tokenizer.encode_plus(" " + t, return_offsets_mapping=True,
                                                                   add_special_tokens=False).offset_mapping for t in
                                        input_texts]
                    log = {
                        "correct": fever_prediction['label'].lower() == fever_prediction['predicted_label'].lower(),
                        "evidence_correct": evidence_correct if class_labels[0] != gt_dict["NOT ENOUGH INFO"] else None,
                        "claim": claim,
                        "label": fever_prediction['label'],
                        "predicted_label": fever_prediction['predicted_label'],
                        "evidence": fever_prediction['evidence'],
                        "predicted_evidence": fever_prediction['predicted_evidence'],
                        "input_sentences": (input_texts, sentences_in_input_texts, titles_in_input_texts),
                        "input_sentence_scores": input_sentence_scores,
                        "is_multihop": is_multihop,
                        "is_multihop_crossarticle": is_multihop_article_sample,
                        "linear_combinations_per_provenance": linear_combinations_per_provenance,
                        "metadata": metadata[0]
                    }
                    log['input_sentences_perword_probs'] = sentence_perword_logprobs.exp()
                    log['input_text_offset_encodings'] = offset_encodings
                    logs.append(log)
                elif self.config.get("log_predictions_conflicting_evidence", False):
                    log_ex.update({
                        "correct": fever_prediction['label'].lower() == fever_prediction['predicted_label'].lower(),
                        "evidence_correct": evidence_correct if class_labels[0] != gt_dict["NOT ENOUGH INFO"] else None,
                        "claim": claim,
                        "input_sentences": (input_texts, sentences_in_input_texts, titles_in_input_texts),
                        "input_sentence_scores": input_sentence_scores,
                        "is_multihop": is_multihop,
                        "is_multihop_crossarticle": is_multihop_article_sample,
                        "metadata": metadata[0]
                    })
                    self.write_conflictingevidence_log(log_ex, csvwriter)

            online_loss = sum(losslist) / len(losslist)
            if reranking_only:
                it.set_description(
                    f"val Loss: {online_loss:.3f}, R@5 {online_r5:.3f}")
            else:
                online_acc = sum(p == l for p, l in zip(preds, gts)) / len(preds)
                it.set_description(
                    f"val Loss: {online_loss:.3f} ACC: {online_acc:.3f}, R@5 {online_r5:.3f}")

        if self.distributed:
            dist_losslist = cat_lists(
                share_list(losslist, rank=self.global_rank, world_size=self.world_size))
            dist_recall5_hits = cat_lists(
                share_list([recallat5_hits], rank=self.global_rank, world_size=self.world_size))
            dist_recall5_samples = cat_lists(
                share_list([recallat5_samples], rank=self.global_rank, world_size=self.world_size))
            dist_multihop_mask = cat_lists(
                share_list(multihop_mask, rank=self.global_rank, world_size=self.world_size))

            if self.config.get("dump_topk_relevances_predictionscores", False):
                dist_relevance_scores = cat_lists(
                    share_list(relevance_scores, rank=self.global_rank, world_size=self.world_size))
                dist_prediction_scores = cat_lists(
                    share_list(prediction_scores, rank=self.global_rank, world_size=self.world_size))
                if self.global_rank == 0:
                    with open("topk_rs_ps_FEVER_dev.json", "w") as f:
                        json.dump({"relevance_scores": dist_relevance_scores,
                                   "prediction_scores": dist_prediction_scores}, f)

            loss = sum(dist_losslist) / len(dist_losslist)
            positive_recall = sum(dist_recall5_hits) / sum(dist_recall5_samples)

            logger.info(f"S: {get_model(model).training_steps} Validation Loss: {loss}")
            logger.info(
                f"Recall@5 for S/R classes: {positive_recall} ({sum(dist_recall5_hits)} / {sum(dist_recall5_samples)})")
            if self.config.get("paragraph_supervision", False):
                dist_recall5_paragraph_hits = cat_lists(
                    share_list([paragraph_recall5_hits], rank=self.global_rank, world_size=self.world_size))
                positive_paragraph_recall = sum(dist_recall5_paragraph_hits) / sum(dist_recall5_samples)
                logger.info(
                    f"Paragraph Recall@5 for S/R classes: {positive_paragraph_recall} ({sum(dist_recall5_paragraph_hits)} / {sum(dist_recall5_samples)})")
            best_score = positive_recall
            if not reranking_only:
                dist_preds = cat_lists(
                    share_list(preds, rank=self.global_rank, world_size=self.world_size))
                dist_gts = cat_lists(share_list(gts, rank=self.global_rank, world_size=self.world_size))
                dist_fever_preds = cat_lists(
                    share_list(dataset_predictions, rank=self.global_rank, world_size=self.world_size))
                assert len(dist_preds) == len(dist_gts)

                if self.config.get("do_IDK_toprel_eval", False):
                    # score_vector = torch.FloatTensor([item[-1] for item in dist_preds])
                    def neg_entropy(log_p):
                        p = log_p.exp() / log_p.exp().sum()
                        return p @ p.log()

                    score_vector = torch.FloatTensor(
                        [neg_entropy(item[0][:2]) for item in dist_preds])
                    best_threshold, best_accuracy = None, None
                    for threshold in score_vector:
                        NEIs = score_vector < threshold
                        fixed_preds = [2 if is_nei else torch.argmax(item[0][:2]).item() for is_nei, item in
                                       zip(NEIs, dist_preds)]
                        accuracy = sum(p == gt for p, gt in zip(fixed_preds, dist_gts)) / len(dist_gts)
                        if best_accuracy is None or accuracy > best_accuracy:
                            best_threshold = threshold
                            best_accuracy = accuracy
                    _new_preds, _new_fever_preds = [], []

                    gt_list = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
                    for p, fp, s in zip(dist_preds, dist_fever_preds, score_vector):
                        is_nei = s < best_threshold
                        sr_numerical_category = 2 if is_nei else torch.argmax(p[0][:2]).item()
                        fp['predicted_label'] = gt_list[sr_numerical_category]
                        _new_fever_preds.append(fp)
                        _new_preds.append(sr_numerical_category)
                    dist_fever_preds = _new_fever_preds
                    dist_preds = _new_preds
                else:
                    accuracy = sum(p == gt for p, gt in zip(dist_preds, dist_gts)) / len(dist_gts)
                try:
                    strict_score, label_accuracy, precision, recall, f1 = fever_score(dist_fever_preds)
                except ZeroDivisionError as e:
                    strict_score, label_accuracy, precision, recall, f1 = 0, 0, 0, 0, 0
                if self.config.get("score_focus", False) == "recall":
                    best_score = recall
                elif self.config.get("score_focus", False) == "accuracy":
                    best_score = label_accuracy
                else:
                    best_score = strict_score
                if self.global_rank == 0:
                    logger.info(f"Accuracy: {accuracy}")
                    print_eval_stats(predictions=dist_preds, labels=dist_gts)
                    logger.info(f"Official FEVER evaluation:\n"
                                f"Strict score: {strict_score}\n"
                                f"Label accuracy: {label_accuracy}\n"
                                f"Precision/Recall/F1: {precision}/{recall}/{f1}\n")
                    logger.info(f"Fever predictions: {len(dist_fever_preds)}")

                assert len(dist_fever_preds) == len(dist_multihop_mask)

                def filter_mh(l):
                    return [item for item, mask in zip(l, dist_multihop_mask) if mask]

                try:
                    mh_dist_fever_preds = filter_mh(dist_fever_preds)
                    mh_strict_score, mh_label_accuracy, mh_precision, mh_recall, mh_f1 = fever_score(
                        mh_dist_fever_preds)
                except ZeroDivisionError:
                    mh_strict_score, mh_label_accuracy, mh_precision, mh_recall, mh_f1 = 0, 0, 0, 0, 0
                if self.global_rank == 0:
                    print_eval_stats(predictions=filter_mh(dist_preds), labels=filter_mh(dist_gts))
                    logger.info(f"MH Official FEVER evaluation:\n"
                                f"MH Strict score: {mh_strict_score}\n"
                                f"MH Label accuracy: {mh_label_accuracy}\n"
                                f"MH Precision/Recall/F1: {mh_precision}/{mh_recall}/{mh_f1}\n")
                    logger.info(f"MH Fever predictions: {len(mh_dist_fever_preds)}")
        else:
            raise NotImplementedError("Not distributed run is not supported anymore.")

        if log_results:
            self.write_logs(fn, logs)
            del logs

        if not self.config['test_only']:
            if self.config.get("save_all_checkpoints", False) and (not self.distributed or self.global_rank == 0):
                self.save_model(label_accuracy, model, optimizer_dict, positive_recall, reranking_only, strict_score)

            if best_score > self.best_score:
                logger.info(f"{best_score} ---> New BEST!")
                self.best_score = best_score
                improvement = True
                # Saving disabled for now
                if (not self.distributed or self.global_rank == 0) and not self.config.get("save_all_checkpoints",
                                                                                           False):
                    self.save_model(label_accuracy, model, optimizer_dict, positive_recall, reranking_only,
                                    strict_score)
        if self.distributed:
            # do not start training with other processes
            # this is useful when process 0 saves the model,
            # if there is no barrier other processes already do training
            # logger.debug(f"Entering exit barrier R {self.rank}")
            dist.barrier()
            # logger.debug(f"Passed exit barrier R {self.rank}")
        if self.config.get("log_predictions_conflicting_evidence",
                           False) or "dump_topk_notcorrelated_sentences" in self.config:
            log_fhandle.close()

        model = model.train()
        return best_score, improvement

    def get_input_text_sentences(self, fid_shaped_src, passage_special_token, sentence_special_token,
                                 title_special_token):
        input_texts = []
        sentences_in_input_texts = []
        titles_in_input_texts = []
        sentence_id = 0
        start_token = -1
        title_start = None
        for i in range(len(fid_shaped_src)):
            if fid_shaped_src[i] == passage_special_token:
                if title_start:
                    title = self.tokenizer.decode(fid_shaped_src[title_start + 1:i])
                end_token = i
                input_texts.append(fid_shaped_src[start_token + 1:end_token + 1])
                sentence_id += 1
                start_token = i
            elif fid_shaped_src[i] == title_special_token:
                title_start = i
            elif fid_shaped_src[i] == sentence_special_token:
                end_token = i
                input_texts.append(fid_shaped_src[start_token + 1:end_token + 1])
                sentences_in_input_texts.append(sentence_id)
                titles_in_input_texts.append(title)
                sentence_id += 1
                start_token = i
        input_texts = [self.tokenizer.decode(t) for t in input_texts]
        return input_texts, sentences_in_input_texts, titles_in_input_texts

    @torch.no_grad()
    def validate_HOVER(self, model: TransformerClassifier, val_iter, optimizer_dict=None):
        improvement = False
        model = model.eval()
        for param in model.parameters():
            param.grad = None

        recallat5_hits = 0  # not considering NEI
        paragraph_recall5_hits = 0

        recallat5_samples = 0
        multihop_mask = []

        gts, preds = [], []
        dataset_predictions = []

        losslist = []
        passage_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.passage_special_token)
        gt_dict = gt_dicts[self.config["dataset"]]
        annotated_evidence_count = []
        predicted_evidence_count = []

        it = tqdm(enumerate(val_iter), total=ceil(len(val_iter.dataset) / val_iter.batch_size))
        for i, (src, src_type_ids, src_mask, label, metadata) in it:
            if self.config.get("log_total_results", False):
                if i > self.config["log_total_results"]:
                    break
            # This takes out condenser predictions, to check that our validation is correct
            if self.config.get("debug_pred_like_condenser", False):
                condenser_preds = metadata[0]['condenser_predicted']
            # Move to gpu
            src, src_mask = src.to(self.torch_device), src_mask.to(
                self.torch_device)
            if src_type_ids is not None:
                src_type_ids = src_type_ids.to(self.torch_device)

            if self.config.get("disable_token_type_ids", False):
                src_type_ids = []
            assert self.config['dataset'] == "hover"
            assert len(metadata[0]['evidence']) > 0
            multihop_mask.append(metadata[0]['orig_sample']['num_hops'])

            src = src[0]
            src_mask = src_mask[0]
            if src_type_ids is not None:
                src_type_ids = src_type_ids[0]

            loss, validation_outputs = self.forward_pass(src, src_type_ids, src_mask,
                                                         label, metadata, model, validation=True)

            losslist.append(loss.item())

            #######
            # Get predicted evidence
            sentence_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sentence_special_token)
            passage_mask = src_mask.bool().view(-1)
            fid_shaped_src = src.view(-1)[passage_mask]
            fid_shaped_src_list = fid_shaped_src.tolist()
            sentences_at_input = (src == sentence_special_token).sum(-1).cumsum(0)
            sentence_tokens_mask = (fid_shaped_src == sentence_special_token)

            if self.config.get("dump_attention_matrices", False):
                attention_matrix = validation_outputs["attention_weights"]
                self.dump_attention_matrix(attention_matrix, src, fid_shaped_src, sentences_at_input, metadata)

            class_logprobs, class_labels = validation_outputs['marg_logprobs'], validation_outputs['marg_labels']
            output_logits = validation_outputs['output_logits']

            """
            Forward output already contains sentence-level normalized log-probs
            """
            sentence_logprobs, sentence_perword_logprobs, sentence_perword_logits = self.get_logprobs_per_sentence_from_logits(
                output_logits.squeeze(0),
                fid_shaped_src_list,
                passage_special_token,
                sentence_special_token,
                return_perword_logprobs=True,
                return_perword_logits=True)

            linear_combinations_per_provenance = \
                (sentence_perword_logits - sentence_perword_logprobs)[:, 0, 0].exp()
            orig_sentence_logprobs = sentence_logprobs
            if self.config.get("paragraph_supervision", False):
                paragraph_logprobs, paragraph_perword_logprobs = self.get_logprobs_per_sentence_from_logits(
                    output_logits.squeeze(0),
                    fid_shaped_src_list,
                    passage_special_token,
                    sentence_special_token,
                    paragraph_supervision=True,
                    return_perword_logprobs=True)

            """
            #1 Predicting sentence relevance
            
            Code predicts 2 things:
            Predict TOP-5 relevant sentences
            Predict ALL relevant sentences
            """

            if self.config.get("predict_evidence_from_logits", False):
                assert self.config.get("paragraph_supervision", False) == False, "Not tested for HoVer yet"
                """
                Predict according to logits, not according to sentence-normalized probabilities
                Logits are logs of per-sentence probabilities scaled by linear coefficient about final prediction
                """
                logits_per_provenance = torch.logsumexp(sentence_perword_logits, 1)
                if not self.config.get("predict_evidence_according_class", False):
                    logsummed_logits_of_relevance = torch.logsumexp(logits_per_provenance[:, :2], dim=-1)
                    top_relevant_sentence_scores, top_relevant_sentence_indices = torch.topk(
                        logsummed_logits_of_relevance,
                        k=5 * 10)

                    # This should be equivalent to predicting from probs
                    logsummed_logits_per_provenance = \
                        torch.stack((logsummed_logits_of_relevance, logits_per_provenance[:, 2]))
                    m = torch.max(logsummed_logits_per_provenance, -1)
                    relevant_sentence_indices = m.indices == 0
                    relevant_sentence_scores = m.values[relevant_sentence_indices]
                else:
                    predicted_SR_class = torch.argmax(class_logprobs[:2]).item()
                    top_relevant_sentence_scores, top_relevant_sentence_indices = torch.topk(
                        logits_per_provenance[:, predicted_SR_class], k=5 * 10)

                    # This should be equivalent to predicting from probs
                    m = torch.max(logits_per_provenance, -1)
                    relevant_sentence_indices = m.indices == predicted_SR_class
                    relevant_sentence_scores = m.values[relevant_sentence_indices]

            elif self.config.get("predict_evidence_according_class", False):
                assert self.config.get("paragraph_supervision", False) == False
                predicted_SR_class = torch.argmax(class_logprobs[:2]).item()

                top_relevant_sentence_scores, top_relevant_sentence_indices = torch.topk(
                    sentence_logprobs[:, predicted_SR_class], k=5 * 10)
                m = torch.max(sentence_logprobs, -1)
                relevant_sentence_indices = m.indices == predicted_SR_class
                relevant_sentence_scores = m.values[relevant_sentence_indices]

            elif self.config.get("predict_evidence_from_combination", False):
                assert self.config.get("paragraph_supervision", False) == False
                """
                Predict from linear combinations/prediction scores/ extra-degrees-of-freedom/ Ks in the paper
                """
                top_relevant_sentence_scores, top_relevant_sentence_indices = torch.topk(
                    linear_combinations_per_provenance, k=5 * 10)

                # Relevant indices are still predicted from probs
                # The linear coefficients are independent of relevance category
                m = torch.max(sentence_logprobs, -1)
                relevant_sentence_indices = m.indices == predicted_SR_class
                relevant_sentence_scores = m.values[relevant_sentence_indices]
            else:
                # SUM PROBABILITY OF SUPPORTED/NOT_SUPPORTED RELEVANCES
                sentence_logprobs = torch.stack(
                    (torch.logsumexp(sentence_logprobs[:, :2], dim=-1), sentence_logprobs[:, 2]), dim=-1)
                if self.config.get("paragraph_supervision", False):
                    paragraph_logprobs = torch.stack(
                        (torch.logsumexp(paragraph_logprobs[:, :2], dim=-1), paragraph_logprobs[:, 2]), dim=-1)
                # PREDICT FROM COMBINED PROBABILITY OF RELEVANCE
                top_relevant_sentence_scores, top_relevant_sentence_indices = torch.topk(sentence_logprobs[:, 0],
                                                                                         k=5 * 10)
                m = torch.max(sentence_logprobs, -1)
                relevant_sentence_indices = m.indices == 0
                relevant_sentence_scores = m.values[relevant_sentence_indices]
                if self.config.get("paragraph_supervision", False):
                    top_relevant_paragraph_scores, top_relevant_paragraph_indices = paragraph_logprobs[:, 0] \
                        .sort(descending=True)
                    m = torch.max(paragraph_logprobs, -1)
                    relevant_paragraph_indices = m.indices == 0
                    relevant_paragraph_scores = m.values[relevant_paragraph_indices]

            """
            #3 Postprocessing predicted indices
            """
            # FOR PARAGRAPHS
            if self.config.get("paragraph_supervision", False):
                top_relevant_paragraph_indices = top_relevant_paragraph_indices.tolist()
                # convert indices to titles
                predicted_paragraph_titles = [metadata[0]['titles'][i] for i in top_relevant_paragraph_indices][:5]
                gt_title_annotations = [list({title for title, _ in annotation}) for annotation in
                                        metadata[0]['evidence']]
                # compute paragraph hits for accuracy@k metric
                paragraph_hit = any(
                    all(gttitle in predicted_paragraph_titles for gttitle in gt_titles) for gt_titles in
                    gt_title_annotations)
                paragraph_recall5_hits += int(paragraph_hit)

            # FOR SENTENCES
            relevant_sentence_indices = [si for si, s in enumerate(relevant_sentence_indices) if s]
            top_relevant_sentence_indices = top_relevant_sentence_indices.tolist()

            ## DEBUG
            if self.config.get("DEBUG_CHEAT_ON_VAL_BY_PREDEXACTREL", False):
                relevant_sentence_indices = top_relevant_sentence_indices[:len(metadata[0]['relevant_sentence_labels'])]

            # record statistics
            predicted_evidence_count.append(len(relevant_sentence_indices))
            annotated_evidence_count.append(len(metadata[0]['relevant_sentence_labels']))

            sentences_at_input = sentences_at_input.tolist()
            top_deduplicated_scores, top_evidences = self.get_predicted_evidence(metadata,
                                                                                 top_relevant_sentence_indices,
                                                                                 top_relevant_sentence_scores,
                                                                                 sentences_at_input)
            predicted_deduplicated_scores, predicted_evidences = self.get_predicted_evidence(metadata,
                                                                                             relevant_sentence_indices,
                                                                                             relevant_sentence_scores,
                                                                                             sentences_at_input)

            top_evidences = [list(e) for e in top_evidences]
            predicted_evidences = [list(e) for e in predicted_evidences]

            #### WRITE PREDICTION INTO JSON FORMAT

            sr_numerical_category = torch.argmax(class_logprobs).item()
            gt_numerical_category = gt_dict[label[0]]

            gt_list = ["SUPPORTED", "NOT_SUPPORTED"]
            preds.append(sr_numerical_category)
            gts.append(gt_numerical_category)

            hover_prediction = {
                "label": label[0],
                "predicted_label": gt_list[sr_numerical_category],
                "example": metadata[0]['orig_sample'],
                "top_predicted_evidence": top_evidences,
                "all_predicted_evidence": predicted_evidences,
            }
            if self.config.get("debug_pred_like_condenser", False):
                pid_titles_map = self.db.get_all(table="documents", columns=["pid", "document_title"],
                                                 column_name="pid", column_value=[p[0] for p in condenser_preds],
                                                 fetch_all=True)
                pid_titles_map = dict(pid_titles_map)
                try:
                    hover_prediction['top_predicted_evidence'] = hover_prediction[
                        'all_predicted_evidence'] = [(pid_titles_map[p[0]], p[1]) for p in condenser_preds]
                except IndexError as e:
                    logger.error(f"NOT FOUND ID!!!! {condenser_preds}")
                    raise e

            dataset_predictions.append(hover_prediction)

            def is_evidence_correct(ground_truth_evidence, predicted_evidence, max_evidence=5):
                return all([actual_sent in predicted_evidence[:max_evidence] for actual_sent in
                            ground_truth_evidence])

            evidence_correct = is_evidence_correct(metadata[0]['evidence'], top_evidences,
                                                   max_evidence=max([5, len(metadata[0]['evidence'])]))
            recallat5_hits += int(evidence_correct)
            recallat5_samples += 1

            assert len(label) == 1

            online_r5 = recallat5_hits / recallat5_samples if recallat5_samples > 0 else 1.
            assert len(label) == 1
            if log_results or self.config.get("log_predictions_conflicting_evidence", False):
                assert False, "Not tested / reimplemented for HOVER"
                input_texts, sentences_in_input_texts, titles_in_input_texts = self.get_input_text_sentences(
                    fid_shaped_src, passage_special_token, sentence_special_token, title_special_token)
                # Assume global log_softmax + per-sentence softmax
                with torch.cuda.amp.autocast():
                    input_sentence_scores = orig_sentence_logprobs.exp()
                claim = self.tokenizer.decode(src[0][1:src[0].cpu().tolist().index(title_special_token)])
                if log_results:

                    offset_encodings = [self.tokenizer.encode_plus(" " + t, return_offsets_mapping=True,
                                                                   add_special_tokens=False).offset_mapping for t in
                                        input_texts]
                    log = {
                        "correct": hover_prediction['label'].lower() == hover_prediction['predicted_label'].lower(),
                        "evidence_correct": evidence_correct if class_labels[0] != gt_dict["NOT ENOUGH INFO"] else None,
                        "claim": claim,
                        "label": hover_prediction['label'],
                        "predicted_label": hover_prediction['predicted_label'],
                        "evidence": hover_prediction['evidence'],
                        "predicted_evidence": hover_prediction['predicted_evidence'],
                        "input_sentences": (input_texts, sentences_in_input_texts, titles_in_input_texts),
                        "input_sentence_scores": input_sentence_scores,
                        "is_multihop": is_multihop,
                        "is_multihop_crossarticle": is_multihop_article_sample,
                        "linear_combinations_per_provenance": linear_combinations_per_provenance,
                        "metadata": metadata[0]
                    }
                    log['input_sentences_perword_probs'] = sentence_perword_logprobs.exp()
                    log['input_text_offset_encodings'] = offset_encodings
                    logs.append(log)
                elif self.config.get("log_predictions_conflicting_evidence", False):
                    log_ex.update({
                        "correct": hover_prediction['label'].lower() == hover_prediction['predicted_label'].lower(),
                        "evidence_correct": evidence_correct if class_labels[0] != gt_dict["NOT ENOUGH INFO"] else None,
                        "claim": claim,
                        "input_sentences": (input_texts, sentences_in_input_texts, titles_in_input_texts),
                        "input_sentence_scores": input_sentence_scores,
                        "is_multihop": is_multihop,
                        "is_multihop_crossarticle": is_multihop_article_sample,
                        "metadata": metadata[0]
                    })
                    self.write_conflictingevidence_log(log_ex, csvwriter)

            online_loss = sum(losslist) / len(losslist)
            online_acc = sum(p == l for p, l in zip(preds, gts)) / len(preds)

            avg_ann = sum(annotated_evidence_count) / len(annotated_evidence_count)
            avg_pred = sum(predicted_evidence_count) / len(predicted_evidence_count)
            _, _, em, _ = hover_eval(dataset_predictions)
            _, _, positives_em, _ = hover_eval([d for d in dataset_predictions if d['label'] == "SUPPORTED"])
            _, _, negatives_em, _ = hover_eval([d for d in dataset_predictions if d['label'] != "SUPPORTED"])
            it.set_description(
                f"val Loss: {online_loss:.3f} ACC: {online_acc:.3f}, R@5 {online_r5:.3f}, A/P evidence count "
                f"{avg_ann:.1f}/{avg_pred:.1f}, EM: {em:.1f}, POS EM {positives_em:.1f}, NEG EM {negatives_em:.1f}")

        if self.distributed:
            # BASIC EVALUATION
            dist_losslist = cat_lists(
                share_list(losslist, rank=self.global_rank, world_size=self.world_size))
            dist_recall5_hits = cat_lists(
                share_list([recallat5_hits], rank=self.global_rank, world_size=self.world_size))
            dist_recall5_samples = cat_lists(
                share_list([recallat5_samples], rank=self.global_rank, world_size=self.world_size))
            dist_multihop_mask = cat_lists(
                share_list(multihop_mask, rank=self.global_rank, world_size=self.world_size))

            loss = sum(dist_losslist) / len(dist_losslist)
            positive_recall = sum(dist_recall5_hits) / sum(dist_recall5_samples)

            logger.info(f"S: {get_model(model).training_steps} Validation Loss: {loss}")
            logger.info(
                f"Recall@5: {positive_recall} ({sum(dist_recall5_hits)} / {sum(dist_recall5_samples)})")
            # PARAGRAPH EVALUATION, IF NEEDED
            if self.config.get("paragraph_supervision", False):
                dist_recall5_paragraph_hits = cat_lists(
                    share_list([paragraph_recall5_hits], rank=self.global_rank, world_size=self.world_size))
                positive_paragraph_recall = sum(dist_recall5_paragraph_hits) / sum(dist_recall5_samples)
                logger.info(
                    f"Paragraph Recall@5 for S/R classes: {positive_paragraph_recall} ({sum(dist_recall5_paragraph_hits)} / {sum(dist_recall5_samples)})")
            dist_preds = cat_lists(
                share_list(preds, rank=self.global_rank, world_size=self.world_size))
            dist_gts = cat_lists(share_list(gts, rank=self.global_rank, world_size=self.world_size))
            dist_hover_preds = cat_lists(
                share_list(dataset_predictions, rank=self.global_rank, world_size=self.world_size))

            if self.config.get("DEBUG_SAVE_PREDS", False):
                if self.global_rank == 0:
                    with open("predictions_hover_dbg.pkl", "wb") as wf:
                        pickle.dump(dist_hover_preds, wf)

            assert len(dist_preds) == len(dist_gts)

            # accuracy = sum(p == gt for p, gt in zip(dist_preds, dist_gts)) / len(dist_gts)
            # with open("predictions_hover.pkl", "wb") as f:
            #     pickle.dump(dist_hover_preds, f)
            hover_score, label_accuracy, em, f1 = hover_eval(dist_hover_preds)
            if self.config.get("score_focus", False) == "recall":
                best_score = recall
            elif self.config.get("score_focus", False) == "em":
                best_score = em
            elif self.config.get("score_focus", False) == "accuracy":
                best_score = label_accuracy
            else:
                best_score = hover_score
            if self.global_rank == 0:
                print_eval_stats(predictions=dist_preds, labels=dist_gts)
                logger.info(f"Overall HoVer evaluation:\n"
                            f"HoVer score: {hover_score}\n"
                            f"Label accuracy: {label_accuracy}\n"
                            f"Sentence EM: {em}\n"
                            f"Sentence F1: {f1}\n")
                logger.info(f"HoVer predictions: {len(dist_hover_preds)}")

                assert len(dist_hover_preds) == len(dist_multihop_mask)

                def filter_mh(l, hops):
                    return [item for item, mask in zip(l, dist_multihop_mask) if mask == hops]

                for hops in [2, 3, 4]:
                    try:
                        mh_dist_hover_preds = filter_mh(dist_hover_preds, hops)
                        mh_hover_score, mh_label_accuracy, mh_em, mh_f1 = hover_eval(
                            mh_dist_hover_preds)
                    except ZeroDivisionError:
                        mh_hover_score, mh_label_accuracy, mh_precision, mh_recall, mh_f1 = 0, 0, 0, 0, 0
                        mh_dist_hover_preds = []
                    print_eval_stats(predictions=filter_mh(dist_preds, hops), labels=filter_mh(dist_gts, hops))
                    logger.info(f"MH{hops} Evaluation \n"
                                f"MH{hops} HoVer score: {mh_hover_score}\n"
                                f"MH{hops} Label accuracy: {mh_label_accuracy}\n"
                                f"MH{hops} Sentence EM: {mh_em}\n"
                                f"MH{hops} Sentence F1: {mh_f1}\n")
                    logger.info(f"MH{hops} HoVer predictions: {len(mh_dist_hover_preds)}")
        else:
            assert False, "Not tested / reimplemented for HOVER"

        if log_results:
            assert False, "Not tested / reimplemented for HOVER"
            self.write_logs(fn, logs)
            del logs

        if not self.config['test_only']:
            if self.config.get("save_all_checkpoints", False) and (not self.distributed or self.global_rank == 0):
                self.save_model(label_accuracy, model, optimizer_dict, positive_recall, False, hover_score)

            if best_score > self.best_score:
                logger.info(f"{best_score} ---> New BEST!")
                self.best_score = best_score
                improvement = True
                # Saving disabled for now
                if (not self.distributed or self.global_rank == 0) and not self.config.get("save_all_checkpoints",
                                                                                           False):
                    self.save_model(label_accuracy, model, optimizer_dict, positive_recall, False,
                                    hover_score)
        if self.distributed:
            # do not start training with other processes
            # this is useful when process 0 saves the model,
            # if there is no barrier other processes already do training
            # logger.debug(f"Entering exit barrier R {self.rank}")
            dist.barrier()
            # logger.debug(f"Passed exit barrier R {self.rank}")
        if self.config.get("log_predictions_conflicting_evidence", False):
            log_fhandle.close()

        model = model.train()
        return best_score, improvement

    @torch.no_grad()
    def validate_FAVIQ(self, model: TransformerClassifier, val_iter, optimizer_dict=None,
                       log_results=False):
        improvement = False
        model = model.eval()
        for param in model.parameters():
            param.grad = None

        recallat5_hits = 0
        recallat5_samples = 0

        gts, preds = [], []

        online_r5 = 0.

        losslist = []
        passage_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.passage_special_token)
        title_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.title_special_token)

        if log_results:
            import xlsxwriter
            from colour import Color
            mkdir(".results")
            fn = f".results/faviq_lrm_verifier_{log_results}"
            if self.distributed:
                fn += f"_{self.global_rank}"

            logs = []
        gt_dict = {
            "SUPPORTS": 0,
            "REFUTES": 1
        }

        it = tqdm(enumerate(val_iter), total=ceil(len(val_iter.dataset) / val_iter.batch_size))
        for i, (src, src_type_ids, src_mask, label, metadata) in it:
            if self.config.get("log_total_results", False):
                if i > self.config["log_total_results"]:
                    break
            # Move to gpu
            src, src_mask = src.to(self.torch_device), src_mask.to(
                self.torch_device)
            if src_type_ids is not None:
                src_type_ids = src_type_ids.to(self.torch_device)
            if self.config.get("disable_token_type_ids", False):
                src_type_ids = []

            src = src[0]
            src_mask = src_mask[0]
            if src_type_ids is not None:
                src_type_ids = src_type_ids[0]

            loss, validation_outputs = self.forward_pass(src, src_type_ids, src_mask, label, metadata, model,
                                                         validation=True)

            losslist.append(loss.item())

            #######
            # Get predicted evidence
            sentence_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sentence_special_token)
            passage_mask = src_mask.bool().view(-1)
            fid_shaped_src = src.view(-1)[passage_mask]
            fid_shaped_src_list = fid_shaped_src.tolist()

            class_logits, class_labels = validation_outputs['marg_logprobs'], validation_outputs['marg_labels']
            output_logits = validation_outputs['output_logits']

            """
            Forward output already contains sentence-level normalized log-probs
            """
            sentence_logprobs, sentence_perword_logprobs, sentence_perword_logits = self.get_logprobs_per_sentence_from_logits(
                output_logits.squeeze(0),
                fid_shaped_src_list,
                passage_special_token,
                sentence_special_token,
                return_perword_logprobs=True,
                return_perword_logits=True)

            orig_sentence_logprobs = sentence_logprobs
            paragraph_logprobs, paragraph_perword_logprobs, paragraph_perword_logits = self.get_logprobs_per_sentence_from_logits(
                output_logits.squeeze(0),
                fid_shaped_src_list,
                passage_special_token,
                sentence_special_token,
                paragraph_supervision=True,
                return_perword_logprobs=True,
                return_perword_logits=True)
            orig_paragraph_logprobs = paragraph_logprobs
            linear_combinations_per_provenance = \
                (paragraph_perword_logits - paragraph_perword_logprobs)[:, 0, 0].exp()

            # Default
            paragraph_logprobs = torch.stack(
                (torch.logsumexp(paragraph_logprobs[:, :2], dim=-1), paragraph_logprobs[:, 2]), dim=-1)

            relevant_paragraph_scores, relevant_paragraph_indices = \
                paragraph_logprobs[:, 0].sort(descending=True)

            predicted_paragraph_indices = []
            if metadata[0]['relevant_passage_labels'] != []:
                relevant_paragraph_indices = relevant_paragraph_indices.tolist()

                gt_indices = set(metadata[0]['relevant_passage_labels'])
                predicted_paragraph_indices = relevant_paragraph_indices[:len(gt_indices)]

                paragraph_hits = [predtitle in gt_indices for predtitle in predicted_paragraph_indices]
                paragraph_hit = sum(paragraph_hits) / len(paragraph_hits)
                recallat5_hits += paragraph_hit
                recallat5_samples += 1
                online_r5 = recallat5_hits / recallat5_samples

            sr_numerical_category = torch.argmax(class_logits).item()
            gt_numerical_category = gt_dict[label[0]]

            preds.append(sr_numerical_category)
            gts.append(gt_numerical_category)

            assert len(label) == 1

            online_loss = sum(losslist) / len(losslist)
            online_acc = sum(p == l for p, l in zip(preds, gts)) / len(preds)
            it.set_description(
                f"val Loss: {online_loss:.3f} ACC: {online_acc:.3f}, R@5 {online_r5:.3f}")

            # Save data for result logging
            if log_results:
                input_text_ids = []
                for s in src.tolist():
                    psg_index = s.index(passage_special_token)
                    i_txt = s[psg_index + 1:]
                    last_sentence_idx = i_txt[::-1].index(sentence_special_token)
                    i_txt = i_txt[:-last_sentence_idx]
                    input_text_ids.append(
                        [t for t in i_txt if t not in [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id,
                                                       self.tokenizer.sep_token_id]])
                input_texts = [self.tokenizer.decode(t) for t in input_text_ids]
                # Assume global log_softmax + per-sentence softmax
                with torch.cuda.amp.autocast():
                    input_paragraph_scores = orig_paragraph_logprobs.exp()
                claim = self.tokenizer.decode(src[0][1:src[0].cpu().tolist().index(title_special_token)])
                offset_encodings = [self.tokenizer.encode_plus(t, return_offsets_mapping=True,
                                                               add_special_tokens=False).offset_mapping for t in
                                    input_texts]
                log = {
                    "correct": gt_numerical_category == sr_numerical_category,
                    "evidence_correct": paragraph_hits if metadata[0]['relevant_passage_labels'] != [] else "",
                    "claim": claim,
                    "label": gt_numerical_category,
                    "predicted_label": sr_numerical_category,
                    "evidence": list(gt_indices) if metadata[0]['relevant_passage_labels'] != [] else [],
                    "predicted_evidence": list(predicted_paragraph_indices),
                    "input_paragraphs": (input_texts, metadata[0]['titles']),
                    "input_paragraphs_scores": input_paragraph_scores,
                    "linear_combinations_per_provenance": linear_combinations_per_provenance,
                    "metadata": metadata[0],
                    "input_paragraphs_perword_probs": paragraph_perword_logprobs.exp(),
                    "input_text_offset_encodings": offset_encodings
                }
                logs.append(log)

        if self.distributed:
            dist_losslist = cat_lists(
                share_list(losslist, rank=self.global_rank, world_size=self.world_size))
            dist_recall5_hits = cat_lists(
                share_list([recallat5_hits], rank=self.global_rank, world_size=self.world_size))
            dist_recall5_samples = cat_lists(
                share_list([recallat5_samples], rank=self.global_rank, world_size=self.world_size))

            loss = sum(dist_losslist) / len(dist_losslist)
            positive_recall = sum(dist_recall5_hits) / sum(dist_recall5_samples)

            logger.info(f"S: {get_model(model).training_steps} Validation Loss: {loss}")
            logger.info(
                f"Recall@5 for S/R classes: {positive_recall} ({sum(dist_recall5_hits):.2f} / {sum(dist_recall5_samples)})")
            dist_preds = cat_lists(
                share_list(preds, rank=self.global_rank, world_size=self.world_size))
            dist_gts = cat_lists(share_list(gts, rank=self.global_rank, world_size=self.world_size))
            assert len(dist_preds) == len(dist_gts)
            label_accuracy = sum(p == gt for p, gt in zip(dist_preds, dist_gts)) / len(dist_gts)

            if self.config.get("score_focus", False) == "recall":
                best_score = positive_recall
            else:
                best_score = label_accuracy
            if self.global_rank == 0:
                logger.info(f"Accuracy: {label_accuracy}")
                print_eval_stats(predictions=dist_preds, labels=dist_gts)
                if "model" in self.config and self.config.get("write_predictions_faviq", False):
                    FILE = self.config["model"] + "_results.json"
                    logger.info("Writing out: " + FILE)
                    with open(FILE, "w") as f:
                        json.dump([dist_preds, dist_gts], f)
                logger.info(f"Recall:{positive_recall}\n")
                logger.info(f"Predictions: {len(dist_preds)}")

        if log_results and not self.distributed or (log_results and self.distributed and self.global_rank == 0):
            # with open("DUMP_PICKLE_DBG.pkl", "wb") as f:
            #     pickle.dump(logs, f)
            # exit()
            # with open("DUMP_PICKLE_DBG.pkl", "rb") as f:
            #     logs = pickle.load(f)
            self.write_faviq_logs(fn, logs)
            del logs
            exit()
        if not self.config['test_only']:
            if self.config.get("save_all_checkpoints", False) and (not self.distributed or self.global_rank == 0):
                self.save_model(label_accuracy, model, optimizer_dict, positive_recall, False, None)

            if best_score > self.best_score:
                logger.info(f"{best_score} ---> New BEST!")
                self.best_score = best_score
                improvement = True
                # Saving disabled for now
                if (not self.distributed or self.global_rank == 0) and not self.config.get("save_all_checkpoints",
                                                                                           False):
                    self.save_model(label_accuracy, model, optimizer_dict, positive_recall, False, None)
        if self.distributed:
            # do not start training with other processes
            # this is useful when process 0 saves the model,
            # if there is no barrier other processes already do training
            # logger.debug(f"Entering exit barrier R {self.rank}")
            dist.barrier()
            # logger.debug(f"Passed exit barrier R {self.rank}")

        model = model.train()
        return best_score, improvement

    def get_predicted_evidence(self, metadata, relevant_sentence_indices, relevant_sentence_scores, sentences_at_input):
        predicted_evidence = []
        deduplicated_scores = []
        for si, ss in zip(relevant_sentence_indices, relevant_sentence_scores):
            passage_index = np.searchsorted(sentences_at_input, si, side='right')
            passage_title = metadata[0]['titles'][passage_index]
            sent_ranges = metadata[0]['sent_ranges'][passage_index].split('|')
            correction_offset = sentences_at_input[passage_index - 1] if passage_index > 0 else 0
            assert correction_offset >= 0
            sentence_index = sent_ranges[si - correction_offset]
            sentence_index = int(sentence_index)
            # make sure no evidence is predicted twice
            if (passage_title, sentence_index) not in predicted_evidence:
                predicted_evidence.append((passage_title, sentence_index))
                deduplicated_scores.append(ss)
        return deduplicated_scores, predicted_evidence

    def save_model(self, label_accuracy, model, optimizer_dict, positive_recall, reranking_only, strict_score):
        serializable_model_name = self.config['verifier_transformer_type'].replace("/", "_")
        dataset_name = self.config["dataset"]
        saveable_model = get_model(model)
        saveable_model.optimizer_state_dict = optimizer_dict

        # Note that model training is fully resumable
        # it contains .optimizer_state_dict and .training_steps (=number of updates)
        if reranking_only:
            saved_name = os.path.join(self.config['save_dir'], f"{dataset_name}_verifier_"
                                                               f"R{positive_recall:.4sf}_"
                                                               f"B_{self.config['block_size']}_"
                                                               f"S{get_model(model).training_steps}_"
                                                               f"M{serializable_model_name}_"
                                                               f"{get_timestamp()}_{socket.gethostname()}")
        else:
            strict_score_string = f"S{strict_score:.4f}_" if strict_score is not None else ""
            saved_name = os.path.join(self.config['save_dir'], f"{dataset_name}_verifier_"
                                                               f"A{label_accuracy:.4f}_"
                                                               f"{strict_score_string}"
                                                               f"B_{self.config['block_size']}_"
                                                               f"S{get_model(model).training_steps}_"
                                                               f"M{serializable_model_name}_"
                                                               f"{get_timestamp()}_{socket.gethostname()}")
        self.last_ckpt_name = saved_name
        device_map = None
        if hasattr(saveable_model, "model_parallel") \
                and saveable_model.model_parallel:
            device_map = saveable_model.device_map
            model.deparallelize()
        torch.save(saveable_model, saved_name)
        if device_map is not None:
            saveable_model.parallelize(device_map)

    def make_parallel(self, model):
        """
        Wrap model in dataparallel, if possible
        """

        model.to(self.local_rank)
        if self.config.get("multi_gpu", False) and torch.cuda.device_count() > 1:
            model = DataParallel(model)
            logger.info("DataParallel active!")
            logger.info(f"Using device ids: {model.device_ids}")

        if self.distributed:
            assert type(model) != DataParallel
            # model = DDP(model, device_ids=[self.local_rank],find_unused_parameters=True)
            model = DDP(model, device_ids=[self.local_rank])
            # torch.autograd.set_detect_anomaly(True)

        return model

    def write_faviq_logs(self, fn, logs):
        HEADER = ["ExampleID", "Correct", "EvidenceCorrect", "Claim", "Predicted", "GT", "PredictedEvidence",
                  "GTEvidence",
                  "GT_sentences", "Predicted_Support",
                  "Predicted_Refute"]

        import xlsxwriter
        from colour import Color
        workbook = xlsxwriter.Workbook(f"{fn}.xlsx")
        worksheet = workbook.add_worksheet()

        colors = []
        # GRADIENT FROM WHITE TO BLUE
        COLOR_GRADIENT_FROM = Color("#000000")
        COLOR_GRADIENT_TO = Color("#002eff")
        COLOR_RESOLUTION = 1000
        colors.append(list(COLOR_GRADIENT_FROM.range_to(COLOR_GRADIENT_TO, COLOR_RESOLUTION)))

        # GRADIENT FROM WHITE TO RED
        COLOR_GRADIENT_FROM = Color("#000000")
        COLOR_GRADIENT_TO = Color("#ff002e")
        COLOR_RESOLUTION = 1000
        colors.append(list(COLOR_GRADIENT_FROM.range_to(COLOR_GRADIENT_TO, COLOR_RESOLUTION)))

        # GRADIENT FROM WHITE TO YELLOW
        COLOR_GRADIENT_FROM = Color("#000000")
        COLOR_GRADIENT_TO = Color("#ffd100")
        COLOR_RESOLUTION = 1000
        colors.append(list(COLOR_GRADIENT_FROM.range_to(COLOR_GRADIENT_TO, COLOR_RESOLUTION)))

        def get_color_idx(colors, selected_att, max_att):
            total_colors = len(colors)

            # scale colors linearly from 0 to max
            return round((selected_att / max_att) * (total_colors - 1))

        # WRITEOUT header
        for column, h in enumerate(HEADER):
            worksheet.write(0, column, h)

        for row, log in tqdm(enumerate(logs, start=1)):
            input_texts, titles_in_input_texts = log["input_paragraphs"]
            gt_texts = [input_texts[i] for i in log['evidence']]

            simple_row = [
                log["metadata"]['id'],
                log['correct'],
                log['evidence_correct'],
                log['claim'],
                log['predicted_label'],
                log['label'],
                log['predicted_evidence'],
                log['evidence'],
                " ".join(gt_texts)
            ]
            # "input_sentences": (input_texts, sentences_in_input_texts),
            # "input_sentence_scores": input_sentence_scores,
            for column, v in enumerate(simple_row):
                worksheet.write(row, column, str(v))

            input_paragraph_scores = log["input_paragraphs_scores"]
            linear_combinations_per_provenance = log["linear_combinations_per_provenance"]
            # Max-normalize, so there won't be larger number than 1 here
            linear_combinations_per_provenance = linear_combinations_per_provenance / linear_combinations_per_provenance.max()

            topk = input_paragraph_scores.topk(10, 0)  # values, indices

            input_paragraphs_perword_probs = log["input_paragraphs_perword_probs"]
            offset_encodings = log['input_text_offset_encodings']

            max_relevance_score = input_paragraph_scores.max().item()
            max_prediction_score = linear_combinations_per_provenance.max().item()

            default_format = workbook.add_format({'font_color': 'black',
                                                  'text_wrap': True})
            for i in range(topk.indices.shape[1] - 1):  # for SUPPORT/REFUTE
                title_opts = []
                pred_opts = []
                formats_lists = []
                for top_index in topk.indices[:, i]:
                    relevance_score = input_paragraph_scores[top_index, i]
                    prediction_score = linear_combinations_per_provenance[top_index]

                    color_opt = {'font_color': colors[i][
                        get_color_idx(colors[i], relevance_score.item(), max_relevance_score)].get_hex_l(),
                                 'text_wrap': True}
                    title_opts.append(color_opt)

                    color_opt = {'font_color': colors[i][
                        get_color_idx(colors[i], prediction_score.item(), max_prediction_score)].get_hex_l(),
                                 'text_wrap': True}
                    pred_opts.append(color_opt)

                    formats_per_item = []
                    for word_score in input_paragraphs_perword_probs[top_index, :, i]:
                        color_opt = {'font_color': colors[i][
                            get_color_idx(colors[i], word_score.item(),
                                          input_paragraphs_perword_probs[top_index, :, i].max().item())].get_hex_l(),
                                     'text_wrap': True}
                        formats_per_item.append(workbook.add_format(color_opt))
                    formats_lists.append(formats_per_item)

                boldformats = [workbook.add_format({
                    'font_color': y['font_color'],
                    'bold': True,
                    'text_wrap': True
                }) for y in title_opts]

                boldformats_prediction = [workbook.add_format({
                    'font_color': y['font_color'],
                    'bold': True,
                    'text_wrap': True
                }) for y in pred_opts]

                # write cell
                column = i + len(simple_row)
                data = []
                error_occured = False
                for relevance_score, idx, formats, boldformat, boldformat_prediction in zip(
                        topk.values[:, i], topk.indices[:, i],
                        formats_lists,
                        boldformats, boldformats_prediction):
                    data += [boldformat, titles_in_input_texts[idx] + f"_RS({relevance_score.item():.2f})"]
                    data += [boldformat_prediction,
                             f"_PS[{linear_combinations_per_provenance[idx].item():.2f}]\n" + " -- "]
                    current_sentence_offset_encodings = offset_encodings[idx]  # except for sentence token
                    if not len(formats) + input_texts[idx].count("<sentence>") >= len(
                            current_sentence_offset_encodings) and not error_occured:
                        error_occured = True
                        logger.warning("Tokenization error occured!")
                    sentence_tokens = 0
                    for enc_idx, (s, e) in enumerate(current_sentence_offset_encodings):
                        try:
                            if (input_texts[idx])[s:e] == "<sentence>":
                                data += [default_format, (input_texts[idx])[s:e] + "\n"]
                                sentence_tokens += 1
                            elif input_texts[idx][s:e] == "":
                                pass
                            else:
                                data += [formats[enc_idx - sentence_tokens], (input_texts[idx])[s:e] + " "]
                        except IndexError as e:
                            logger.error(f"Indexing error occured for\n{input_texts[idx]};{s}:{e}")
                    data[-1] = "\n"
                data = tuple(data)
                worksheet.write_rich_string(row, column, *data)
        workbook.close()

    def write_logs(self, fn, logs):
        HEADER = ["ExampleID", "Correct", "EvidenceCorrect", "Claim", "Predicted", "GT", "PredictedEvidence",
                  "GTEvidence",
                  "Is_MultiHop", "Is_Multihop_Crossarticle", "GT_sentences", "Predicted_Support",
                  "Predicted_Refute", "Predicted_NEI"]

        import xlsxwriter
        from colour import Color
        workbook = xlsxwriter.Workbook(f"{fn}.xlsx")
        worksheet = workbook.add_worksheet()

        colors = []
        # GRADIENT FROM WHITE TO BLUE
        COLOR_GRADIENT_FROM = Color("#000000")
        COLOR_GRADIENT_TO = Color("#002eff")
        COLOR_RESOLUTION = 1000
        colors.append(list(COLOR_GRADIENT_FROM.range_to(COLOR_GRADIENT_TO, COLOR_RESOLUTION)))

        # GRADIENT FROM WHITE TO RED
        COLOR_GRADIENT_FROM = Color("#000000")
        COLOR_GRADIENT_TO = Color("#ff002e")
        COLOR_RESOLUTION = 1000
        colors.append(list(COLOR_GRADIENT_FROM.range_to(COLOR_GRADIENT_TO, COLOR_RESOLUTION)))

        # GRADIENT FROM WHITE TO YELLOW
        COLOR_GRADIENT_FROM = Color("#000000")
        COLOR_GRADIENT_TO = Color("#ffd100")
        COLOR_RESOLUTION = 1000
        colors.append(list(COLOR_GRADIENT_FROM.range_to(COLOR_GRADIENT_TO, COLOR_RESOLUTION)))

        def get_color_idx(colors, selected_att, max_att):
            total_colors = len(colors)

            # scale colors linearly from 0 to max
            return round((selected_att / max_att) * (total_colors - 1))

        # WRITEOUT header
        for column, h in enumerate(HEADER):
            worksheet.write(0, column, h)

        for row, log in tqdm(enumerate(logs, start=1)):
            input_texts, sentences_in_input_texts, titles_in_input_texts = log["input_sentences"]

            # only select scored input sentences
            input_texts = [input_texts[i] for i in sentences_in_input_texts]

            if "relevant_sentence_labels" in log['metadata']:
                gt_indices = []
                each_passage_sentence_count = [len(x.split("|")) for x in log['metadata']['sent_ranges']]
                for passage_idx, sent_idx in log['metadata']['relevant_sentence_labels']:
                    flat_sentence_index = sum(each_passage_sentence_count[:passage_idx]) + sent_idx
                    gt_indices.append(flat_sentence_index)
                gt_texts = [input_texts[i] for i in gt_indices]
            else:
                assert len(input_texts) == len(log['metadata']['sentence_labels'])
                gt_texts = [f"({label})-{text}" for text, label in zip(input_texts, log['metadata']['sentence_labels'])
                            if
                            label in ['supporting', 'refuting']]

            simple_row = [
                log["metadata"]['id'],
                log.get("correct", ""),
                log.get("evidence_correct", ""),
                log.get("claim", ""),
                log.get("predicted_label", ""),
                log.get("label", ""),
                log.get("predicted_evidence", ""),
                log.get("evidence", ""),
                log.get('is_multihop', ""),
                log.get('is_multihop_crossarticle', ""),
                "\n".join(gt_texts)
            ]
            # "input_sentences": (input_texts, sentences_in_input_texts),
            # "input_sentence_scores": input_sentence_scores,
            for column, v in enumerate(simple_row):
                worksheet.write(row, column, str(v))

            input_sentence_scores = log["input_sentence_scores"]
            linear_combinations_per_provenance = log["linear_combinations_per_provenance"]
            # Max-normalize, so there won't be larger number than 1 here
            linear_combinations_per_provenance = linear_combinations_per_provenance / linear_combinations_per_provenance.max().abs()

            topk = input_sentence_scores.topk(min(10, input_sentence_scores.shape[0]), 0)  # values, indices

            input_sentences_perword_probs = log["input_sentences_perword_probs"]
            offset_encodings = [log['input_text_offset_encodings'][i] for i in sentences_in_input_texts]

            max_sentence_score = input_sentence_scores.max().item()
            max_prediction_score = linear_combinations_per_provenance.max().item()
            for i in range(topk.indices.shape[1]):  # for SUPPORT/REFUTE/NEI
                title_opts = []
                pred_opts = []
                sentence_formats_lists = []
                for top_index in topk.indices[:, i]:
                    relevance_score = input_sentence_scores[top_index, i]
                    prediction_score = linear_combinations_per_provenance[top_index]

                    color_opt = {'font_color': colors[i][
                        get_color_idx(colors[i], relevance_score.item(), max_sentence_score)].get_hex_l(),
                                 'text_wrap': True}
                    title_opts.append(color_opt)

                    color_opt = {'font_color': colors[i][
                        get_color_idx(colors[i], prediction_score.item(), max_prediction_score)].get_hex_l(),
                                 'text_wrap': True}
                    pred_opts.append(color_opt)

                    formats_per_sentence = []
                    for word_score in input_sentences_perword_probs[top_index, :, i]:
                        color_opt = {'font_color': colors[i][
                            get_color_idx(colors[i], word_score.item(),
                                          input_sentences_perword_probs[top_index, :, i].max().item())].get_hex_l(),
                                     'text_wrap': True}
                        formats_per_sentence.append(workbook.add_format(color_opt))
                    sentence_formats_lists.append(formats_per_sentence)

                boldformats = [workbook.add_format({
                    'font_color': y['font_color'],
                    'bold': True,
                    'text_wrap': True
                }) for y in title_opts]

                boldformats_prediction = [workbook.add_format({
                    'font_color': y['font_color'],
                    'bold': True,
                    'text_wrap': True
                }) for y in pred_opts]

                # write cell
                column = i + len(simple_row)
                data = []
                error_occured = False
                for relevance_score, idx, sentence_formats, boldformat, boldformat_prediction in zip(
                        topk.values[:, i], topk.indices[:, i],
                        sentence_formats_lists,
                        boldformats, boldformats_prediction):
                    data += [boldformat, titles_in_input_texts[idx] + f"_RS({relevance_score.item():.2f})"]
                    data += [boldformat_prediction,
                             f"_PS[{linear_combinations_per_provenance[idx].item():.2f}]\n" + " -- "]
                    current_sentence_offset_encodings = offset_encodings[idx][:-1]  # except for sentence token
                    if not len(sentence_formats) >= len(current_sentence_offset_encodings) and not error_occured:
                        error_occured = True
                        logger.warning("Tokenization error occured!")
                    for enc_idx, (s, e) in enumerate(current_sentence_offset_encodings):
                        try:
                            data += [sentence_formats[enc_idx], (" " + input_texts[idx])[s:e]]
                        except IndexError as e:
                            pass
                    data[-1] = "\n"
                data = tuple(data)
                worksheet.write_rich_string(row, column, *data)
        workbook.close()

    @torch.no_grad()
    def validate_interpretability_FEVER(self, model, val_iter):
        model = model.eval()
        for param in model.parameters():
            param.grad = None

        passage_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.passage_special_token)
        title_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.title_special_token)
        sentence_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sentence_special_token)

        mkdir(".results")
        fn = f".results/lrm_verifier_interpretability_"
        if self.distributed:
            fn += f"_{self.global_rank}"

        logprobs = []
        logs = []
        gt_dict = {
            "SUPPORTS": 0,
            "REFUTES": 1,
            "NOT ENOUGH INFO": 2
        }

        it = tqdm(enumerate(val_iter), total=ceil(len(val_iter.dataset) / val_iter.batch_size))
        for i, (src, src_type_ids, src_mask, label, metadata) in it:
            if self.config.get("log_total_results", False):
                if i > self.config["log_total_results"]:
                    break
            assert metadata[0]['id'] in self.word_level_evidence
            annotation = self.word_level_evidence[metadata[0]['id']]
            if len(annotation['label']) == 0:  # this happened in v0.1 dataset
                continue
            # Move to gpu
            src, src_type_ids, src_mask = src.to(self.torch_device), src_type_ids.to(
                self.torch_device), src_mask.to(self.torch_device)
            src = src[0]
            src_mask = src_mask[0]
            if src_type_ids is not None:
                src_type_ids = src_type_ids[0]
            _, validation_outputs = self.forward_pass(src, src_type_ids, src_mask, label, metadata, model,
                                                      validation=True)

            #######
            # Get predicted evidence
            sentence_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sentence_special_token)
            passage_mask = src_mask.bool().view(-1)
            fid_shaped_src = src.view(-1)[passage_mask]
            fid_shaped_src_list = fid_shaped_src.tolist()
            sentences_at_input = (src == sentence_token_id).sum(-1).cumsum(0)
            sentence_tokens_mask = (fid_shaped_src == sentence_token_id)

            if self.config.get("train_masker", False):
                class_logits, class_labels, output_logits = None, None, None
                mask_weights = validation_outputs["mask_weights"]
                mask_weights = mask_weights.view(-1)[passage_mask]
                nei_logprob = validation_outputs['marg_logprobs'][-1].item()
                logprobs.append(-nei_logprob)
                index_per_provenance = []
                start_token = -1
                for i in range(len(fid_shaped_src)):
                    if fid_shaped_src[i] == passage_special_token:
                        start_token = i
                    elif fid_shaped_src[i] == sentence_special_token:
                        index_per_provenance.append(list(range(start_token + 1, i)))
                        start_token = i

                total_provenances = len(index_per_provenance)
                longest_provenance = max(len(x) for x in index_per_provenance)
                padded_linearized_token_logits_perprovenance_idx = flatten(
                    x + [0] * (longest_provenance - len(x)) for x in index_per_provenance)
                sentence_perword_scores = mask_weights[padded_linearized_token_logits_perprovenance_idx].view(
                    total_provenances, longest_provenance).float()
            else:
                class_logits, class_labels = validation_outputs['marg_logprobs'], validation_outputs['marg_labels']
                output_logits = validation_outputs['output_logits']

                """
                    Forward output already contains sentence-level normalized log-probs
                """
                sentence_logprobs, sentence_perword_logprobs = self.get_logprobs_per_sentence_from_logits(
                    output_logits.squeeze(0),
                    fid_shaped_src_list,
                    passage_special_token,
                    sentence_special_token,
                    return_perword_logprobs=True)
                orig_sentence_logprobs = sentence_logprobs

            """
            Get sentence texts
            """
            input_texts = []
            sentences_in_input_texts = []
            titles_in_input_texts = []
            sentence_id = 0
            start_token = -1
            title_start = None
            for i in range(len(fid_shaped_src)):
                if fid_shaped_src[i] == passage_special_token:
                    if title_start:
                        title = self.tokenizer.decode(fid_shaped_src[title_start + 1:i])
                    end_token = i
                    input_texts.append(fid_shaped_src[start_token + 1:end_token + 1])
                    sentence_id += 1
                    start_token = i
                elif fid_shaped_src[i] == title_special_token:
                    title_start = i
                elif fid_shaped_src[i] == sentence_special_token:
                    end_token = i
                    input_texts.append(fid_shaped_src[start_token + 1:end_token + 1])
                    sentences_in_input_texts.append(sentence_id)
                    titles_in_input_texts.append(title)
                    sentence_id += 1
                    start_token = i
            """
            Guess offset encoding for input texts
            """
            input_texts = [self.tokenizer.decode(t) for t in input_texts]
            offset_encodings = [self.tokenizer.encode_plus(" " + t, return_offsets_mapping=True,
                                                           add_special_tokens=False).offset_mapping for t in
                                input_texts]

            gt_indices = []
            each_passage_sentence_count = [len(x.split("|")) for x in metadata[0]['sent_ranges']]
            for passage_idx, sent_idx in metadata[0]['relevant_sentence_labels']:
                flat_sentence_index = sum(each_passage_sentence_count[:passage_idx]) + sent_idx
                gt_indices.append(flat_sentence_index)

            if self.config.get("train_masker", False):
                input_sentence_scores = None
            else:
                # Assume global log_softmax + per-sentence softmax
                with torch.cuda.amp.autocast():
                    input_sentence_scores = orig_sentence_logprobs.exp()
                    input_sentence_scores = input_sentence_scores[gt_indices][:, :2].sum(1)

            claim = self.tokenizer.decode(src[0][1:src[0].cpu().tolist().index(title_special_token)])

            input_texts = [input_texts[si] for idx, si in enumerate(sentences_in_input_texts) if idx in gt_indices]
            offset_encodings = [offset_encodings[si] for idx, si in enumerate(sentences_in_input_texts) if
                                idx in gt_indices]
            titles_in_input_texts = [titles_in_input_texts[idx] for idx in gt_indices]

            # For Relevance prediction
            gt_label = gt_dict[label[0]]
            log = {
                "claim": claim,
                "label": gt_label,
                "input_sentences": (input_texts, titles_in_input_texts),
                # I validated interpretability for supported/refuted class

                "input_sentence_scores": input_sentence_scores.cpu(),
                "metadata": metadata[0],
                "gt_indices": gt_indices
            }
            if self.config.get("word_level_evidence", False):  # backward comp
                log["annotated_text"] = " ".join([annotation['data'][start:end]
                                                  for start, end, _ in annotation['label']])
            else:
                log["annotated_text"] = annotation['label']

            if self.config.get("train_masker", False):
                lp = sentence_perword_scores[gt_indices]
            else:
                # normalize per row into same scale, linearly
                lp = sentence_perword_logprobs.exp()[gt_indices]
                lp = lp[:, :, :2].sum(-1)

            # # linearize,divide by maximum, and reshape back
            # lp_lin = lp.transpose(-1, -2).reshape(-1, lp.shape[1])
            # lp_lin = (lp_lin.T / lp_lin.max(-1).values).T
            # lp = lp_lin.reshape(lp.shape[0], lp.shape[-1], lp.shape[-2]).transpose(-1, -2)

            log['input_sentences_perword_scores'] = lp.tolist()
            log['input_text_offset_encodings'] = offset_encodings
            logs.append(log)

        if self.distributed:
            logs = cat_lists(
                share_list(logs, rank=self.global_rank, world_size=self.world_size))
            if logprobs:
                logprobs = cat_lists(
                    share_list(logprobs, rank=self.global_rank, world_size=self.world_size))

        if logprobs:
            logger.info(f"Validation avg. -logP(C=NEI) loss: {sum(logprobs) / len(logprobs):.2f}")

        all_values = set()
        logger.info(f"Computing F1 from {len(logs)} samples")
        for log in logs:
            all_values = all_values.union(set(flatten(log['input_sentences_perword_scores'])))
        all_values = list(sorted(list(all_values)))

        best_f1 = 0
        best_t = 0
        best_tp, best_fp, best_fn = 0, 0, 0
        pbar = tqdm(all_values)
        if self.config.get("baseline_predict_all", False):
            threshold = -1
            best_f1, best_t, TP, FP, FN = self.eval_wlann_for_threshold(logs, threshold)
        elif self.config.get("baseline_predict_overlaps", False):
            best_f1, best_t = self.eval_wlann_for_overlaps(logs)
        else:
            for threshold in pbar:
                avg_f1, threshold, TP, FP, FN = self.eval_wlann_for_threshold(logs, threshold)
                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    best_t = threshold
                    best_tp, best_fp, best_fn = TP, FP, FN
                    pbar.set_description(f"Best F1 {best_f1:.2f}, best T {best_t:.8f}")

        logger.info(f"Best F1 score {best_f1:.2f}")
        logger.info(f"Best TP-FP-FN  {best_tp} / {best_fp} / {best_fn}")
        logger.info(f"Best threshold: {best_t}")
        return best_f1

    def eval_wlann_for_threshold(self, logs, threshold):
        # compute F1 for every threshold, report best F1 and best threshold
        f1_sum = 0
        TP, FP, FN = 0, 0, 0
        simpletokenizer_tokenize = lambda s: self.simpletokenizer.tokenize(s).words()
        for log in logs:
            # annotated text from processed dataset
            annotated_items = log['annotated_text']
            # boolean mask for every scored token, based on provided threshold, is computed
            predicted_mask = [[x > threshold for x in l] for l in log['input_sentences_perword_scores']]
            predicted_tokens = []
            assert len(log["input_sentences"][0]) == len(log['input_text_offset_encodings']) == len(predicted_mask)
            for idx, (sentence_text, sentence_encodings, predicted_per_sentence_mask) in enumerate(
                    zip(log["input_sentences"][0], log['input_text_offset_encodings'], predicted_mask)):
                # sentence_text contains raw text of the sentence
                # sentence_encodings contains start/end character offsets of every token in the sentence
                # predicted_per_sentence_mask contains boolean mask for every token in the sentence
                sentence_encodings = sentence_encodings[:-1]  # cut for <sentence> token
                predicted_offsets = [x for m, x in zip(predicted_per_sentence_mask, sentence_encodings) if m]
                # whitespace is added in our preprocessing before text, so we also do it here, so offsets match
                predicted_tokens.extend([(" " + sentence_text)[s:e] for s, e, in predicted_offsets])
            predicted = "".join(predicted_tokens)

            # compute F1 against each annotated item, and take the best one
            f1s = list(sorted(
                [f1_score(predicted, annotated, tokenize=simpletokenizer_tokenize, return_tp_fp_fn=True) for annotated
                 in annotated_items], key=lambda x: -x[0]))
            f1_sum += f1s[0][0]

            # add TPs, FPs, FNs to statistics, for later significance testing
            tp, fp, fn = f1s[0][1]
            TP += tp
            FP += fp
            FN += fn
        avg_f1 = f1_sum / len(logs)

        return avg_f1, threshold, TP, FP, FN

    def eval_wlann_for_overlaps(self, logs):
        f1_sum = 0
        for log in logs:
            annotated_text = log['annotated_text']
            predicted = log['claim'][7:-5]  # Skip Special tokens
            f1_sum += max(
                f1_score(predicted, annotated, tokenize=self.tokenizer.tokenize) for annotated in annotated_text)
        avg_f1 = f1_sum / len(logs)
        return avg_f1, None

    def dump_attention_matrix(self, attention_matrix, src, fid_shaped_src, sentences_at_input, metadata):
        # sentence_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sentence_special_token)
        # passage_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sentence_special_token)
        sentences_at_input = sentences_at_input.cpu()
        import matplotlib
        from matplotlib.pyplot import close
        import numpy as np

        matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
        import matplotlib.pyplot as plt  # drawing heat map of attention weights

        font = {'size': 8}

        matplotlib.rc('font', **font)

        def plot_attention(data, X_label=None, Y_label=None, filename="figure.png"):
            '''
              Plot the attention model heatmap
              Args:
                data: attn_matrix with shape [ty, tx]
                X_label: list of size tx, encoder tags
                Y_label: list of size ty, decoder tags
            '''
            fig, ax = plt.subplots(figsize=(100, 30))  # set figure size
            heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.9)

            # Set axis labels
            if X_label != None and Y_label != None:
                # X_label = [x_label for x_label in X_label]
                # Y_label = [y_label for y_label in Y_label]

                xticks = range(0, len(X_label))
                ax.set_xticks(xticks, minor=False)  # major ticks
                ax.set_xticklabels(X_label, minor=False, rotation=90)  # labels should be 'unicode'

                yticks = range(0, len(Y_label))
                ax.set_yticks(yticks, minor=False)
                ax.set_yticklabels(Y_label, minor=False)  # labels should be 'unicode'

                ax.grid(True)

            # Save Figure
            plt.title(u'Attention Matrix')
            fig.savefig(filename, format="png")
            close(fig)

        attention_matrix = attention_matrix.squeeze(0).transpose(-1, -2)

        sentence_dim_top_indices = attention_matrix.sum(0).sum(-1).topk(k=30, dim=0)[1].sort().values
        token_dim_top_indices = attention_matrix.sum(0).sum(0).topk(k=300, dim=0)[1].sort().values
        sentence_dim_top_indices_set = set(sentence_dim_top_indices.tolist())
        token_dim_top_indices_set = set(token_dim_top_indices.tolist())

        # input_texts = []
        # sentences_in_input_texts = []
        # titles_in_input_texts = []
        # sentence_id = 0
        # start_token = -1
        # title_start = None
        # for i in range(len(fid_shaped_src)):
        #     if fid_shaped_src[i] == passage_special_token:
        #         if title_start:
        #             title = self.tokenizer.decode(fid_shaped_src[title_start + 1:i])
        #         end_token = i
        #         input_texts.append(fid_shaped_src[start_token + 1:end_token + 1])
        #         sentence_id += 1
        #         start_token = i
        #     elif fid_shaped_src[i] == title_special_token:
        #         title_start = i
        #     elif fid_shaped_src[i] == sentence_special_token:
        #         end_token = i
        #         input_texts.append(fid_shaped_src[start_token + 1:end_token + 1])
        #         sentences_in_input_texts.append(sentence_id)
        #         titles_in_input_texts.append(title)
        #         sentence_id += 1
        #         start_token = i
        #
        # input_texts = [self.tokenizer.decode(t) for t in input_texts]
        passage_special_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.passage_special_token)
        sentence_special_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sentence_special_token)
        # Prepend token with its Passage Name, Passage Idx and Sentence idx
        texts = []
        token_index = 0
        for passage_id, inp in enumerate(src.tolist()):
            sentence_idx = -1
            sentence_id = "-1"
            title = metadata[0]['titles'][passage_id]
            sent_ranges = metadata[0]['sent_ranges'][passage_id].split('|')
            for i, t in enumerate(inp):
                if t == self.tokenizer.pad_token_id:
                    continue
                if t in [sentence_special_token_id, passage_special_token_id]:
                    sentence_idx += 1
                    if not sentence_idx == len(sent_ranges):
                        sentence_id = sent_ranges[sentence_idx]

                if token_index in token_dim_top_indices_set:
                    texts.append(f"{title}P{passage_id}#{sentence_id} {self.tokenizer._convert_id_to_token(t)}")
                token_index += 1
        assert token_index == len(fid_shaped_src)
        # texts = self.tokenizer.convert_ids_to_tokens(
        #     [t for i, t in enumerate(fid_shaped_src) if i in token_dim_top_indices_set])

        sentence_labels = []
        for si in range(attention_matrix.shape[-1]):
            if si not in sentence_dim_top_indices_set:
                continue
            passage_index = np.searchsorted(sentences_at_input, si, side='right').item()
            passage_title = metadata[0]['titles'][passage_index]
            sent_ranges = metadata[0]['sent_ranges'][passage_index].split('|')
            correction_offset = sentences_at_input[passage_index - 1] if passage_index > 0 else 0
            assert correction_offset >= 0
            sentence_index = sent_ranges[si - correction_offset]
            sentence_labels.append(f"{passage_title}P{passage_index}#{sentence_index}")

        attention_matrix_filtered = attention_matrix[:, sentence_dim_top_indices][:, :, token_dim_top_indices]
        for i in range(len(attention_matrix_filtered)):
            plot_attention(attention_matrix_filtered[i].cpu(), texts, sentence_labels,
                           filename=f".am/am_{metadata[0]['id']}_{i}.png")

    def get_abias_loss(self, mhweights, fid_shaped_src, metadata):
        mh_weights = torch.stack(mhweights, 0).mean(0).mean(0)  # average over layers and over heads

        each_passage_sentence_count = [len(x.split("|")) for x in metadata['sent_ranges']]
        relevant_passage_indices = []
        gt_indices = []
        for passage_idx, sent_idx in metadata['relevant_sentence_labels']:
            assert each_passage_sentence_count[passage_idx] > sent_idx
            flat_sentence_index = sum(each_passage_sentence_count[:passage_idx]) + sent_idx
            gt_indices.append(flat_sentence_index)
            relevant_passage_indices.append(passage_idx)
        if not len(set(relevant_passage_indices)) > 1:
            return 0.

        pairs = []
        for pi, x in enumerate(relevant_passage_indices):
            for pj, y in enumerate(relevant_passage_indices):
                if x != y and [pj, pi] not in pairs:
                    pairs.append([pi, pj])

        passage_special_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.passage_special_token)
        sentence_special_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sentence_special_token)

        sentence_attention_ranges = []
        start_token = -1
        for i in range(len(fid_shaped_src)):
            if fid_shaped_src[i] == passage_special_token_id:
                start_token = i
            elif fid_shaped_src[i] == sentence_special_token_id:
                sentence_attention_ranges.append([start_token + 1, i])
                start_token = i
        # assert len(sentence_attentions) == mh_weights.shape[1]
        # For loop implementation
        loss = 0.
        for p in pairs:
            p0range = sentence_attention_ranges[p[0]]
            p1range = sentence_attention_ranges[p[1]]

            A = mh_weights[p0range[0]:p0range[1]][:, p[1]].mean()
            B = mh_weights[p1range[0]:p1range[1]][:, p[0]].mean()
            loss += - ((A + B) / 2.).log()

        return loss

    @torch.no_grad()
    def _predict(self, model, test_iter, outfile):
        model = model.eval()
        fever_predictions = []
        passage_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.passage_special_token)

        it = tqdm(enumerate(test_iter), total=ceil(len(test_iter.dataset) / test_iter.batch_size))
        for i, (src, src_type_ids, src_mask, label, metadata) in it:
            # Move to gpu
            src, src_mask = src.to(self.torch_device), src_mask.to(
                self.torch_device)
            if src_type_ids is not None:
                src_type_ids = src_type_ids.to(self.torch_device)

            # This is done for RoBERTa
            if self.config.get("disable_token_type_ids", False):
                src_type_ids = []

            src = src[0]
            src_mask = src_mask[0]
            if src_type_ids is not None:
                src_type_ids = src_type_ids[0]

            _, validation_outputs = self.forward_pass(src, src_type_ids, src_mask,
                                                      metadata=metadata, model=model, validation=True)

            #######
            # Get predicted evidence
            sentence_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sentence_special_token)
            passage_mask = src_mask.bool().view(-1)
            fid_shaped_src = src.view(-1)[passage_mask]
            fid_shaped_src_list = fid_shaped_src.tolist()
            sentences_at_input = (src == sentence_special_token).sum(-1).cumsum(0)
            sentence_tokens_mask = (fid_shaped_src == sentence_special_token)

            class_logits = validation_outputs['marg_logprobs']
            output_logits = validation_outputs['output_logits']

            """
            Forward output already contains sentence-level normalized log-probs
            """
            sentence_logprobs, sentence_perword_logprobs = self.get_logprobs_per_sentence_from_logits(
                output_logits.squeeze(0),
                fid_shaped_src_list,
                passage_special_token,
                sentence_special_token,
                return_perword_logprobs=True)
            orig_sentence_logprobs = sentence_logprobs

            if not self.config.get("predict_evidence_according_class", False):
                if self.config.get("paper_baseline", False):
                    sentence_logprobs = torch.logsumexp(sentence_logprobs, dim=-1).unsqueeze(-1)
                else:
                    sentence_logprobs = torch.stack(
                        (torch.logsumexp(sentence_logprobs[:, :2], dim=-1), sentence_logprobs[:, 2]), dim=-1)
            else:
                marg_sentence_logprobs = torch.stack(
                    (torch.logsumexp(sentence_logprobs[:, :2], dim=-1), sentence_logprobs[:, 2]), dim=-1)

            if self.config.get("predict_top5_sentences", False):
                if self.config.get("predict_evidence_according_class", False):
                    predicted_SR_class = torch.argmax(class_logits).item()
                    if predicted_SR_class == 2:
                        relevant_sentence_scores, relevant_sentence_indices = torch.topk(
                            marg_sentence_logprobs[:, 0], k=5 * 10)
                    else:
                        relevant_sentence_scores, relevant_sentence_indices = torch.topk(
                            sentence_logprobs[:, predicted_SR_class], k=5 * 10)
                else:
                    relevant_sentence_scores, relevant_sentence_indices = torch.topk(sentence_logprobs[:, 0],
                                                                                     k=5 * 10)
            else:
                relevant_sentence_indices = ~ torch.argmax(sentence_logprobs, -1).bool()

            if not self.config.get("predict_top5_sentences", False):
                relevant_sentence_indices = [si for si, s in enumerate(relevant_sentence_indices) if s]
            else:
                relevant_sentence_indices = relevant_sentence_indices.tolist()

            sentences_at_input = sentences_at_input.tolist()
            deduplicated_scores, predicted_evidence = self.get_predicted_evidence(metadata, relevant_sentence_indices,
                                                                                  relevant_sentence_scores,
                                                                                  sentences_at_input)

            predicted_evidence = [list(e) for e in predicted_evidence]
            #### end of  getting predicted evidence
            if self.config.get("predict_top5_sentences", False):
                predicted_evidence = predicted_evidence[:5]
                assert len(predicted_evidence) == 5
                predicted_evidence_scores = deduplicated_scores[:5]

            sr_numerical_category = torch.argmax(class_logits).item()

            gt_list = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

            fever_prediction = {
                "id": metadata[0]['id'],
                "predicted_label": gt_list[sr_numerical_category],
                "predicted_evidence": predicted_evidence,
            }
            fever_predictions.append(fever_prediction)
        with jsonlines.open(outfile, "w") as wf:
            for e in fever_predictions:
                wf.write(e)

    def get_persentencescoresum_L2reg(self, all_passage_tokens_logits, fid_shaped_src_list,
                                      passage_special_token, sentence_special_token):
        output_classes = 3

        token_logits_per_single_provenance_idx = []
        token_logits_per_provenances_idx = []
        start_token = -1
        for i in range(len(fid_shaped_src_list)):
            if fid_shaped_src_list[i] == passage_special_token:
                start_token = i
                if token_logits_per_single_provenance_idx:
                    token_logits_per_provenances_idx.append(token_logits_per_single_provenance_idx)
                    token_logits_per_single_provenance_idx = []
            elif fid_shaped_src_list[i] == sentence_special_token:
                token_logits_per_single_provenance_idx.append(list(range(start_token + 1, i)))
                start_token = i
        # add last paragraph
        assert token_logits_per_single_provenance_idx
        token_logits_per_provenances_idx.append(token_logits_per_single_provenance_idx)
        token_logits_per_single_provenance_idx = []

        token_logits_per_provenance_idx_ = []
        sentence_provenance_indices = []
        for provenance_idx, token_logits_per__provenance_idx in enumerate(token_logits_per_provenances_idx):
            for token_logits_per_sentence_idx in token_logits_per__provenance_idx:
                token_logits_per_provenance_idx_.append(token_logits_per_sentence_idx)
                sentence_provenance_indices.append(provenance_idx)
        token_logits_per_provenance_idx = token_logits_per_provenance_idx_

        total_provenances = len(token_logits_per_provenance_idx)
        longest_provenance = max(len(x) for x in token_logits_per_provenance_idx)

        padded_linearized_token_logits_perprovenance_idx = flatten(
            x + [0] * (longest_provenance - len(x)) for x in token_logits_per_provenance_idx)
        token_logits_perprovenance = all_passage_tokens_logits[padded_linearized_token_logits_perprovenance_idx].view(
            total_provenances, longest_provenance, output_classes).float()

        token_logits_perprovenance_ = token_logits_perprovenance
        token_logits_perprovenance_[torch.isinf(token_logits_perprovenance_)] = 0
        return torch.logsumexp(token_logits_perprovenance_.view(token_logits_perprovenance_.shape[0], -1), -1) \
            .square().sum() / token_logits_perprovenance_.shape[0]

    def write_conflictingevidence_log(self, log, csvwriter):
        input_texts, sentences_in_input_texts, titles_in_input_texts = log["input_sentences"]

        # only select scored input sentences
        input_texts = [input_texts[i] for i in sentences_in_input_texts]

        gt_indices = []
        each_passage_sentence_count = [len(x.split("|")) for x in log['metadata']['sent_ranges']]
        for passage_idx, sent_idx in log['metadata']['relevant_sentence_labels']:
            flat_sentence_index = sum(each_passage_sentence_count[:passage_idx]) + sent_idx
            gt_indices.append(flat_sentence_index)
        gt_texts = [input_texts[i] for i in gt_indices]

        input_sentence_scores = log["input_sentence_scores"]
        topk = input_sentence_scores.topk(10, 0)  # values, indices

        titles = [[], []]
        texts = [[], []]
        for i in range(topk.indices.shape[1] - 1):  # for SUPPORT/REFUTE/NEI
            for idx in topk.indices[:, i]:
                title = titles_in_input_texts[idx]
                text = input_texts[idx]

                titles[i].append(title)
                texts[i].append(text.replace(" <sentence>", ""))

        csvwriter.writerow([
            log['metadata']['id'],
            str(topk.values[:, :2].tolist()),
            str(titles),
            str(texts),
            str(gt_texts),
            str(log['label']),
            str(log['predicted_label']),
            str(log['evidence']),
            str(log['predicted_evidence']),
            str(log['correct']),
            str(log['evidence_correct']),
            str(log['claim']),
            str(log['is_multihop']),
            str(log['is_multihop_crossarticle']),
        ])
