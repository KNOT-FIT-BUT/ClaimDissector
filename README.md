# Introduction

This is the official repository accompanying our work on __Claim-Dissector: An Interpretable Fact-Checking System with
Joint Re-ranking and Veracity Prediction__.

```bibtex
@inproceedings{fajcik-etal-2023-CD,
    title = "{C}laim-{D}issector: An Interpretable Fact-Checking System with Joint Re-ranking and Veracity Prediction",
    author = "Fajcik, Martin and Motlicek, Petr and Smrz, Pavel",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = {https://mfajcik.github.io/assets/pdf/ClaimDissector__V2_public.pdf},
}
```

# Contact us

This is the first version of released code. More comprehensive script documentation and code refactoring can still
happen in future revisions.
Be sure to contact us if you need any help with code navigation.  
Corresponding author `martin.fajcik@vut.cz`.

### Planned TODOs

1. REFACTOR: We haven't touched the code much since the experiments, just verified it works. Once the time window
   appears,
   we hope to refactor the code and make it more readable. For now we follow the idea,
   _"it is better to have messy and working code released, than having clean and non-working code hidden on our disks"_.
2. ~~Release SOTA results on the [REALFC dataset](https://arxiv.org/abs/2104.00640). We will add this to appendices of
   camera-ready.
   The REALFC code is already in the repo.~~ DONE
3. Release checkpoints for each dataset, not just FEVER.
4. Re-test recipe for document retrieval on FEVER.

# Installation

Clone this repository and go to the project root directory.

## Getting the Data

Download and unzip [data](https://nextcloud.fit.vutbr.cz/s/QRCjCeio5gjpgtQ) in the project's root directory.
Download link is protected with password.  
By inputting the password `iagreewithdatasetlicenses`,  __you agree with licensing terms (if present) of datasets__ 
[FEVER](https://github.com/awslabs/fever/blob/master/LICENSE), [FAVIQ](https://github.com/faviq/faviq), [HOVER](https://hover-nlp.github.io/), [REALFC](https://github.com/CambridgeNLIP/verification-real-world-info-needs) and this work's license.
Furthermore, as these datasets were scraped from Wikipedia, you also agree with Wikipedia's licensing, as stated below.

> These data annotations incorporate material from Wikipedia, which is licensed pursuant to the [Wikipedia Copyright Policy](https://en.wikipedia.org/wiki/Wikipedia:Copyrights). These annotations are made available under the license terms described on the applicable Wikipedia article pages, or, where Wikipedia license terms are unavailable, under the Creative Commons Attribution-ShareAlike License (version 3.0), available at <a href="http://creativecommons.org/licenses/by-sa/3.0/">http://creativecommons.org/licenses/by-sa/3.0/</a> (collectively, the “License Terms”). You may not use these files except in compliance with the applicable License Terms.

The following data are available in unzipped folder `.data` (truncated for brevity).
```
.
├── .data
│   ├── FAVIQ
│   │   ├── eval_dpr
│   │   │   ├── faviq_dev.jsonl
│   │   │   └── faviq_test.jsonl
│   │   ├── evidentiality_dpr # contains relevant paragraph annotations from Asai et al. 2021
│   │   │   └── faviq_train_w_evidentiality.jsonl
│   ├── FEVER
│   │   ├── baseline_data # contains data prepared for bm25 processing
│   │   │   ├── get_data.sh
│   │   │   └── processed
│   │   │       ├── qrels.paragraph.shared_task_dev.txt
│   │   │       ├── qrels.paragraph.shared_task_test.txt
│   │   │       ├── ... <truncated>
│   │   │       ├── queries.paragraph.shared_task_dev.tsv
│   │   │       ├── queries.paragraph.shared_task_test.tsv
│   │   │       ├── ... <truncated>
│   │   ├── fever_interleaved_retrieval # contains outputs of bm25 interleaved with athene UKP retrieval
│   │   │   ├── prepared_data_from_run.fever-sentence-top-200.train_ids.tsv
│   │   │   ├── prepared_data_from_run.fever-sentence-top-200.train_texts.tsv
│   │   │   ├── run.fever-paragraph.shared_task_dev.tsv
│   │   │   ├── ... <truncated>
│   │   ├── shared_task_dev.jsonl
│   │   ├── shared_task_test.jsonl
│   │   ├── train.jsonl
│   │   └── verifier
│   │       └── article_level
│   │           ├── shared_task_dev_ret_baseline.jsonl
│   │           ├── shared_task_test_ret_baseline.jsonl
│   │           ├── train_ret_baseline.jsonl
│   │           ├── train_ret_baseline_shuffled_first75p.jsonl
│   │           └── train_ret_baseline_shuffled_last25p.jsonl
│   ├── HOVER
│   │   ├── baleen_retrieval # Inputs for retrieval from Khattab et al.
│   │   │   ├── dev
│   │   │   │   ├── qas.jsonl
│   │   │   │   └── questions.tsv
│   │   │   ├── test
│   │   │   │   ├── qas.jsonl
│   │   │   │   └── questions.tsv
│   │   │   └── train
│   │   │       ├── qas.jsonl
│   │   │       └── questions.tsv
│   │   ├── official
│   │   │   ├── hover_dev_release_v1.1.json
│   │   │   ├── hover_test_release_v1.1.json
│   │   │   └── hover_train_release_v1.1.json
│   │   └── verifier
│   │       └── article_level # Outputs of baleen's retrieval
│   │           ├── baleen_inference_100x4_ordered
│   │           │   ├── output_separate_deduplists_dev.json
│   │           │   ├── output_separate_deduplists_test.json
│   │           │   └── output_separate_deduplists_train.json
│   │           └── baleen_inference_25x4_ordered
│   │               ├── output_separate_deduplists_dev.json
│   │               ├── output_separate_deduplists_test.json
│   │               └── output_separate_deduplists_train.json
│   ├── license.html
│   └── REALFC
│       ├── dev.jsonl
│       ├── test.jsonl
│       └── train.jsonl
├── .checkpoints # should you need a specific checkpoint from our experiments, contact us, and we may scrape our disks and look for it.
│   ├── verifier_A0.7998_S0.7652_R0.9145_B_500_S12500_M.saved_pretrained_debertav3_base_mnli_2022-06-02_00:23_acn66.karolina.it4i.cz
│   └── verifier_A0.8076_S0.7803_R0.9326_B_500_S8500_M.saved_pretrained_debertav3_large_mnli_2022-04-29_23:00_acn01.karolina.it4i.cz
├── .index
│   ├── FEVER_wikipages
│   │   ├── feverwiki_blocks_500.db
│   │   └── fever_wiki_entities_per_sentence.pkl
│   └── HOVER_wikipages
│       └── hoverwiki_blocks_500.db
└── .saved_pretrained
    ├── debertav3_base_mnli
    │   ├── ... <truncated>
    └── debertav3_large_mnli
        ├── ... <truncated>

29 directories, 83 files
```

### Pretrained Checkpoints

We provide two checkpoints from FEVER. The base model checkpoint is trained with K=35, the large model checkpoint is
trained with K=70. To use these checkpoints, you will need to change paths in the runfile config.
See __runfile config guidelines__ at the end of this document for more information.

## Environment

Using `python 3.9.5`, use `requirements.txt` for installation of dependencies

```bash
python -m pip install -r requirements.txt
```

You also need to install `fever-scorer` from [here](https://github.com/sheffieldnlp/fever-scorer). In our
case, `setup.py` had undefined variable `license`. Thus be sure to fix `setup.py` before installing it (it can carry any
string) with `pip install .`.

Before running the program, set locale to en_US

```bash
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
```
### Running scripts
Now you can run the `runefile` scripts, for example, to train the CD model for FEVER conditioned on 35 blocks, you can
run

```
python -m src.fact_checking.verifier.runfile.run_debertamnli_V2_mhsent_noresidual 35
```

Note that in first run, dataset preprocessing is done and cached automatically. This will take a while.
If possible run this script first time on server with many CPU cores. Caching includes (1) preprocessing dataset into
token-ids, ready to be used at model's input;
(2) computing the line offsets of (1) for fast line-level file access (saved in separate `.pkl` file).

If something happens during preprocessing, this may leave your cache unfinished. __You will need to delete them
manually, and rebuild them again__.
By default, cache are located in `.data/[DATASET]/verifier/preprocessed`" folder. Be sure to delete both,
`.jsonl` and `.pkl` files corresponding to your run settings.

Training is optimized to single-node multi-gpu training. If the machine contains multiple CUDA-capable machines, it will
run training on all devices available in `CUDA_VISIBLE_DEVICES`.

Remember to set `"gradient_checkpointing": True,` in the respective file, if you are running out of GPU memory.

### Inspecting Token-level Relevances in XLSX file
By including following options in the runfile config, 
you can inspect token-level relevances in 100 examples randomly selected from validation set.
The outputs are written into .results folder (e.g. here `.results/lrm_verifier_deberta-base_fever_0.xlsx`).
```python
{
    "log_results": "deberta-base_fever",
    "log_total_results": 100,
    "shuffle_validation_set": True,
}
```
An example of such a script that dumps visualized relevances in XLSX file is 
`src/fact_checking/verifier/runfile/fever/eval_debertamnli_V2_mhsent_noresidual_logresults.py`.

The semantics of this file follow [this example](shorturl.at/beTY2) (published in the camera-ready paper version).


## TLR-FEVER

Check out TLR-FEVER's [README](.data/TLR_FEVER/README.md) for more information.

## Runfile Config Guidelines

Each action in this code is executed via "runfile" --- a piece of python code that contains experiment configuration,
and executes distributed training. Please note that the configuration options are experimental, parameters combinations
other than ones in the runfiles provided might not work.

Taking config from `src/fact_checking/verifier/runfile/run_debertamnli_V2_mhsent_noresidual_NOCLS.py` as an example:

```python
config = {
    # type of `transformers` tokenizer
    "verifier_tokenizer_type": "microsoft/deberta-v3-base",
    # path to pretrained `transformers` model. Pretraining is done with src/fact_checking/verifier/pretraining/run_mnli.sh
    "verifier_transformer_type": ".saved_pretrained/debertav3_base_mnli",
    # max length of input sequence to the model   
    "verifier_max_input_length": 500,

    "save_dir": ".saved",  # where the checkpoints will be saved
    "results": ".results",  # where validation results will be saved

    # whether to run only evaluation (True), or also training (False). True in eval_ scripts.
    "test_only": False,
    # always kept to 1, won't work for higher values
    "validation_batch_size": 1,
    # number of steps after which the model is validated on validation set during training
    "validate_after_steps": 500,

    # context length of the model
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

    # cache where the transformers library will save the models
    "transformers_cache": ".Transformers_cache",
    # dataset, one of "fever", "hover", "faviq", or "realfc"
    "dataset": "fever",

    # whether to run training in FP16 mode
    "fp16": True,

    # ddp settings, tuned for Karolina HPC
    "ddp": True,
    "ddp_backend": "nccl",
    "world_size": torch.cuda.device_count(),
    "max_num_threads": 36,

    # size of retrieved blocks, 500-token blocks are preindexed in the downloaded data
    "block_size": 500,

    # random, golden_passage_only, non_golden_passage_only, equal_golden_nongolden_mixture, jiangetal_sup
    "sentence_relevance_negative_ss": "jiangetal_sup",

    # This will create a special version of the dataset, with no NEI examples (as they contain no relevant evidence)
    "reranking_only": False,

    # Weights for loss term
    "lossterm_weights": {
        "cls_loss": 0.,  # ablation, so we keep it 0.
        "sent_loss": 1.,
        "marg_loss": 0.,
    },

    # stategy for predicting relevant sentences
    # to maximize FEVER score, predict top-5
    "predict_top5_sentences": True,

    # use exactly same negative sentences in reranking as Jiang et al. (2020)
    "jiangetal_sup": {
        "ids": "retrieved_data/fever_merged/prepared_data_from_run.fever-sentence-top-200.train_ids.tsv",
        "texts": "retrieved_data/fever_merged/prepared_data_from_run.fever-sentence-top-200.train_texts.tsv"
    },

    # if set to False,  support probability + refute probability will be maximized for relevant sentence
    # if set to False, only target veracity class probability will be maximizes for relevant sentences
    "no_logsumexp_relevance": True,

    # whether to use gradient checkpointing
    "gradient_checkpointing": False,

    # L2 penalty loss weight
    "perword_L2": 0.002,

    # Whether to use hyperlink expansion,
    # "expand_entities_in_retrieval": {
    #     # path to file with expanded hyperlinks
    #     "hyperlinks_per_sentence": ".index/FEVER_wikipages/fever_wiki_entities_per_sentence.pkl",
    #     # how many blocks to add to input from hyperlinks
    #     "extra_context_size": 35
    # },

    # The direction of attention in MHSA at the end of model 
    "mh_w_sent_tokens": True,
    # The number of attention layers in MHSA block
    "mh_sent_layers": 1,
    # Whether to use residual connection in this block
    "mh_sent_residual": False,

    # Based on which metric to perform early stopping on
    "score_focus": "recall",
}
```

Note that some hyperparameters (learning rate and so on) are further set later in runfile

```python
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
    "patience": 10
}
best_params["hidden_dropout"] = best_params["attention_dropout"] = best_params["dropout_rate"]
config.update(best_params)
```
