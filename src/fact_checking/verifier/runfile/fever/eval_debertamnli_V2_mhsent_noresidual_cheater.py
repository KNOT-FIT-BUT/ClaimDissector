# Written by Martin Fajcik <martin.fajcik@vut.cz>
#
import logging
import os
import pickle
import sys
import traceback
from random import randint

import torch
import torch.multiprocessing as torch_mp

from src.fact_checking.verifier.lrm_verifier_trainer import LRMVerifierTrainer
from src.common.distributed_utils import setup_ddp_envinit, cleanup_ddp
from src.common.utility import setup_logging, mkdir, set_seed

"""
Evaluates DeBERTa-based checkpoints of Claim-Dissector on validation set with 100% RaI (with "cheating").
"""

BLOCK_SIZE = 500

MODELS = [
    "verifier_A0.7678_S0.6959_R0.8537_B_500_S500_M.saved_pretrained_debertav3_large_mnli_2022-04-24_23:16_acn10.karolina.it4i.cz",
    "verifier_A0.7757_S0.7373_R0.9045_B_500_S1000_M.saved_pretrained_debertav3_large_mnli_2022-04-25_05:01_acn10.karolina.it4i.cz",
    "verifier_A0.7988_S0.7580_R0.9123_B_500_S1500_M.saved_pretrained_debertav3_large_mnli_2022-04-25_10:46_acn10.karolina.it4i.cz",
    "verifier_A0.7998_S0.7612_R0.9194_B_500_S2000_M.saved_pretrained_debertav3_large_mnli_2022-04-25_16:31_acn10.karolina.it4i.cz",
    "verifier_A0.7999_S0.7630_R0.9217_B_500_S2500_M.saved_pretrained_debertav3_large_mnli_2022-04-25_23:06_acn10.karolina.it4i.cz",
    "verifier_A0.8030_S0.7689_R0.9262_B_500_S3000_M.saved_pretrained_debertav3_large_mnli_2022-04-26_04:52_acn10.karolina.it4i.cz",
    "verifier_A0.8041_S0.7759_R0.9306_B_500_S4000_M.saved_pretrained_debertav3_large_mnli_2022-04-26_16:23_acn10.karolina.it4i.cz",
    "verifier_A0.7972_S0.7673_R0.9295_B_500_S4500_M.saved_pretrained_debertav3_large_mnli_2022-04-27_01:45_acn01.karolina.it4i.cz",
    "verifier_A0.8078_S0.7797_R0.9308_B_500_S5000_M.saved_pretrained_debertav3_large_mnli_2022-04-27_07:31_acn01.karolina.it4i.cz",
    "verifier_A0.8004_S0.7700_R0.9307_B_500_S5500_M.saved_pretrained_debertav3_large_mnli_2022-04-27_13:16_acn01.karolina.it4i.cz",
    "verifier_A0.8055_S0.7786_R0.9315_B_500_S6000_M.saved_pretrained_debertav3_large_mnli_2022-04-27_19:02_acn01.karolina.it4i.cz",
    "verifier_A0.8061_S0.7735_R0.9302_B_500_S6273_M.saved_pretrained_debertav3_large_mnli_2022-04-28_09:08_acn01.karolina.it4i.cz",
    "verifier_A0.8074_S0.7777_R0.9334_B_500_S6500_M.saved_pretrained_debertav3_large_mnli_2022-04-28_12:12_acn01.karolina.it4i.cz",
    "verifier_A0.8039_S0.7737_R0.9316_B_500_S7000_M.saved_pretrained_debertav3_large_mnli_2022-04-29_05:41_acn01.karolina.it4i.cz",
    "verifier_A0.8051_S0.7750_R0.9305_B_500_S7500_M.saved_pretrained_debertav3_large_mnli_2022-04-29_11:28_acn01.karolina.it4i.cz",
    "verifier_A0.8040_S0.7767_R0.9324_B_500_S8000_M.saved_pretrained_debertav3_large_mnli_2022-04-29_17:14_acn01.karolina.it4i.cz",
    "verifier_A0.8076_S0.7803_R0.9326_B_500_S8500_M.saved_pretrained_debertav3_large_mnli_2022-04-29_23:00_acn01.karolina.it4i.cz",
    "verifier_A0.8046_S0.7762_R0.9308_B_500_S8773_M.saved_pretrained_debertav3_large_mnli_2022-04-30_02:31_acn01.karolina.it4i.cz",
    "verifier_A0.8029_S0.7753_R0.9308_B_500_S9000_M.saved_pretrained_debertav3_large_mnli_2022-04-30_05:36_acn01.karolina.it4i.cz",
    "verifier_A0.8005_S0.7705_R0.9335_B_500_S9500_M.saved_pretrained_debertav3_large_mnli_2022-04-30_11:22_acn01.karolina.it4i.cz",
    "verifier_A0.8017_S0.7736_R0.9320_B_500_S10000_M.saved_pretrained_debertav3_large_mnli_2022-04-30_17:08_acn01.karolina.it4i.cz",
    "verifier_A0.8021_S0.7726_R0.9329_B_500_S10500_M.saved_pretrained_debertav3_large_mnli_2022-04-30_22:54_acn01.karolina.it4i.cz", ]

config = {

    # LARGE MODEL
    "verifier_tokenizer_type": ".saved_pretrained/debertav3_large_mnli",
    "verifier_transformer_type": ".saved_pretrained/debertav3_large_mnli",
    # "model": ".saved_karolina/verifier_A0.8076_S0.7803_R0.9326_B_500_S8500_M.saved_pretrained_debertav3_large_mnli_2022-04-29_23:00_acn01.karolina.it4i.cz",

    # BASE MODEL
    # "verifier_tokenizer_type": ".saved_pretrained/debertav3_base_mnli",
    # "verifier_transformer_type": ".saved_pretrained/debertav3_base_mnli",
    # "model": "verifier_A0.7967_S0.7647_R0.9201_B_500_S8000_MMoritzLaurer_DeBERTa-v3-base-mnli-fever-anli_2022-04-10_11:36_acn41.karolina.it4i.cz",

    "verifier_max_input_length": 500,

    "save_dir": ".saved",  # where the checkpoints will be saved
    "results": ".results",  # where validation results will be saved

    "test_only": True,

    "validation_batch_size": 1,
    "validate_after_steps": 500,

    "context_length": int(sys.argv[1]),

    ###############################
    # Data
    ###############################
    "data_cache_dir": ".data/FEVER/verifier/preprocessed",
    "train_data": f".data/FEVER/verifier/article_level/train_ret_baseline.jsonl",
    "val_data": ".data/FEVER/verifier/article_level/shared_task_dev_ret_baseline.jsonl",
    "test_data": ".data/FEVER/verifier/article_level/shared_task_dev_ret_baseline.jsonl",
    "pass_database": f".index/FEVER_wikipages/feverwiki_blocks_{BLOCK_SIZE}.db",

    ###############################
    # Optimization hyper-parameters
    ###############################
    "learning_rate": "",
    "adam_eps": 1e-06,
    "batch_size": 1,
    "true_batch_size": 64,
    "max_grad_norm": 1.,
    "weight_decay": "",
    "hidden_dropout": "",
    "attention_dropout": "",

    "make_sure_golden_passages_in_training": True,

    "optimizer": "",  # adam, adamw
    "scheduler": "",  # "linear",  # None, linear, cosine, constant

    ###############################
    # Miscellaneous options
    ###############################
    # if training has been discontinued, it can be resumed
    "resume_training": False,
    # "resume_checkpoint": ".saved/roberta_verifier_R0.7745_B_250_S1000_Mtextattack_roberta-base-MNLI_2022-01-24_21:28_athena20",

    # maximum number of training steps
    "max_steps": "",  # on resuming the resumed update steps are counted too
    "save_threshold": 0.68,  # save up some disk space
    "save_all_checkpoints": True,

    # cache where the transformers library will save the models
    "transformers_cache": ".Transformers_cache",

    "dataset": "fever",

    "fp16": True,

    "ddp": True,
    "ddp_backend": "nccl",
    "world_size": torch.cuda.device_count(),
    "max_num_threads": 36,

    "block_size": BLOCK_SIZE,

    # random, golden_passage_only, non_golden_passage_only, equal_golden_nongolden_mixture, jiangetal_sup
    "sentence_relevance_negative_ss": "jiangetal_sup",

    "reranking_only": False,

    "lossterm_weights": {
        "cls_loss": 0.,
        "sent_loss": 1.,
        "marg_loss": 1.,

    },

    "predict_top5_sentences": True,
    "jiangetal_sup": {
        "ids": ".data/FEVER/fever_interleaved_retrieval/prepared_data_from_run.fever-sentence-top-200.train_ids.tsv",
        "texts": ".data/FEVER/fever_interleaved_retrieval/prepared_data_from_run.fever-sentence-top-200.train_texts.tsv"
    },

    "no_logsumexp_relevance": True,
    "gradient_checkpointing": False,

    "perword_L2": 0.002,

    "expand_entities_in_retrieval": {
        "hyperlinks_per_sentence": ".index/FEVER_wikipages/fever_wiki_entities_per_sentence.pkl",
        "extra_context_size": 35
    },

    "mh_w_sent_tokens": True,

    "mh_sent_layers": 1,
    "mh_sent_residual": False,

    "cheat_on_val": True,
}

logger = logging.getLogger(__name__)

def ddp_verifier_fit(rank, world_size, config, seed, log_folder):
    torch.set_num_threads(config["max_num_threads"])

    print(f"Running DDP process (PID {os.getpid()}) on rank {rank}.")
    set_seed(seed)
    setup_logging(os.path.join(log_folder, "training", f"rank_{rank}"),
                  logpath=".logs/",
                  config_path="configurations/logging.yml")
    backend = config["ddp_backend"]
    best_params = {
        "max_grad_norm": 1.0,
        "weight_decay": 0.0,
        "learning_rate": 5e-06,
        "adam_eps": 1e-08,
        "warmup_steps": 100,
        "dropout_rate": 0.1,
        "optimizer": "adamw",
        "max_steps": 20_000,
        "scheduler": "constant",
        "patience": 6
    }
    best_params["hidden_dropout"] = best_params["attention_dropout"] = best_params["dropout_rate"]
    config.update(best_params)
    mkdir(config["save_dir"])
    mkdir(config["results"])
    mkdir(config["data_cache_dir"])

    original_bs = config["true_batch_size"]
    config["true_batch_size"] = int(config["true_batch_size"] / config["world_size"])
    logger.info(
        f"True batch size adjusted from {original_bs} to {config['true_batch_size']} due to"
        f" distributed world size {config['world_size']}")

    setup_ddp_envinit(rank, world_size, backend)
    try:
        for m in MODELS:
            config["model"] = os.path.join(".saved_old", m)
            if rank > -1:
                framework = LRMVerifierTrainer(config, local_rank=rank, global_rank=rank, seed=seed)
                framework.distributed = True
                framework.global_rank = rank
                framework.world_size = world_size
            else:
                framework = LRMVerifierTrainer(config, 0, 0, seed)
            validation_accuracy, model_path = framework.fit()
    except BaseException as be:
        logging.error(be)
        logging.error(traceback.format_exc())
        raise be
    finally:
        cleanup_ddp()


def run():
    # preprocess data
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath=".logs/",
                  config_path="configurations/logging.yml")

    framework = LRMVerifierTrainer(config, 0, 0, 1234)
    framework.get_data(config)
    del framework
    # return
    torch.set_num_threads(config["max_num_threads"])
    log_folder = setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                               logpath=".logs/",
                               config_path="configurations/logging.yml")
    seed = 1234
    set_seed(seed)
    logger.info(f"Random seed: {seed}")
    WORLD_SIZE = config["world_size"]

    torch_mp.spawn(ddp_verifier_fit,
                   args=(WORLD_SIZE, config, seed, log_folder),
                   nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    run()
