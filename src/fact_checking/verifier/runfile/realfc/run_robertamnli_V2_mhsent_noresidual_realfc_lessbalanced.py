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
Trains deberta model on REALFC, with partial cross-entropy weighting
"""
BLOCK_SIZE = 500
config = {
    "verifier_tokenizer_type": "textattack/roberta-base-MNLI",
    "verifier_transformer_type": "textattack/roberta-base-MNLI",
    "verifier_max_input_length": 500,

    "save_dir": ".saved",  # where the checkpoints will be saved
    "results": ".results",  # where validation results will be saved

    "test_only": False,
    "validation_batch_size": 1,
    "validate_after_steps": 250,

    ###############################
    # Data
    ###############################
    "data_cache_dir": ".data/REALFC/verifier/preprocessed",

    "train_data": ".data/REALFC/train.jsonl",
    "val_data": ".data/REALFC/dev.jsonl",
    "test_data": ".data/REALFC/test.jsonl",

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
    "save_threshold": 0.48,  # save up some disk space

    # cache where the transformers library will save the models
    "transformers_cache": ".Transformers_cache",

    "dataset": "realfc",

    "fp16": False,

    "ddp": True,
    "ddp_backend": "nccl",
    "world_size": torch.cuda.device_count(),
    "max_num_threads": 36,

    "block_size": BLOCK_SIZE,

    "reranking_only": False,

    "lossterm_weights": {
        "cls_loss": 0.,
        "sent_loss": 1.,
        "marg_loss": 1.,
    },

    "predict_top5_sentences": True,

    "no_logsumexp_relevance": True,
    "gradient_checkpointing": False,

    "perword_L2": 0.002,

    "mh_w_sent_tokens": True,
    "mh_sent_layers": 1,
    "mh_sent_residual": False,

    "balance_weights": [31_544 / 13_122, 31_544 / 6_236, 1.],
    # train set statistics {'supported': 13122, 'refuted': 6236,'neutral': 31544,}
}
# apply only half as much balancing as in full weighted cross-entropy
config["balance_weights"] = [(w + 1.) / 2. for w in config["balance_weights"]]

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
        "learning_rate": 5e-06 * 4,
        "adam_eps": 1e-08,
        "warmup_steps": 100,
        "dropout_rate": 0.1,
        "optimizer": "adamw",
        "max_steps": 15_000,
        "scheduler": "constant",
        "patience": 100
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
    # preprocess data without DDP
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath=".logs/",
                  config_path="configurations/logging.yml")

    framework = LRMVerifierTrainer(config, 0, 0, 1234)
    framework.get_data(config)
    del framework
    # return
    torch.set_num_threads(config["max_num_threads"])
    for i in range(1):
        log_folder = setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                                   logpath=".logs/",
                                   config_path="configurations/logging.yml")
        seed = randint(0, 10_000)
        set_seed(seed)
        logger.info(f"Random seed: {seed}")
        WORLD_SIZE = config["world_size"]

        torch_mp.spawn(ddp_verifier_fit,
                       args=(WORLD_SIZE, config, seed, log_folder),
                       nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    run()
