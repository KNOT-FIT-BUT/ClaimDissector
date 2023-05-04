# Written by Martin Fajcik <martin.fajcik@vut.cz>
#
import os
import sys
import torch

from ...common.utility import setup_logging
from .query_encoder import QueryEncoderFramework

if __name__ == "__main__":
    config = {
        # Omit option, if you do not have the file in your split
        # (e.g. if you have only training/test split, comment-out "test_data_file" option here
        # Path to your training data
        "training_data_file": ".data/FEVER/train.jsonl",
        # Path to your validation data
        "validation_data_file": ".data/FEVER/shared_task_dev.jsonl",

        # Output directory, where to save files with retrievan information
        "output_directory": "retrieved_data",

        # Path to your passage embeddings
        "index_path": ".index/feverwiki_bm25_index_anserini",
        # Path to databse containing passages
        "db_path": ".index/FEVER_wikipages/feverwiki.db",

        # How many top-K passage indices to save into the output file
        "topK_extract": 200,
        "batch_size": 32,

        # K in accuracy@K during online evaluation
        # Also affects precalculated hits
        "topK_eval": 200,

        # Where transformers library download its cache
        "model_cache_dir": ".Transformers_cache",

        "search_type": "BM25",
        "use_RM3": False,
        # fb_terms : int
        #     RM3 parameter for number of expansion terms.
        # fb_docs : int
        #     RM3 parameter for number of expansion documents.
        # original_query_weight : float
        #     RM3 parameter for weight to assign to the original query.
        "RM3_parameters": {"fb_terms": 10, "fb_docs": 10, "original_query_weight": 0.5},

        "BM25_parameters": {"k1": 0.6, "b": 0.5},  # Following Jiang et al. https://aclanthology.org/2021.acl-short.51/

        "threads": 4
    }
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath=".logs/",
                  config_path="configurations/logging.yml")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    extract = lambda file: QueryEncoderFramework.extract_retrieval(config, file,
                                                                   os.path.join(config["output_directory"],
                                                                                os.path.basename(f'ret'
                                                                                                 f'_{file}'
                                                                                                 f'_{config["search_type"]}'
                                                                                                 f'_at{config["topK_eval"]}'
                                                                                                 f'.jsonl')))
    extract(config["validation_data_file"])
    extract(config["training_data_file"])

    # Val data recall@1000
    #  95.064
    #  93.689 + RM3 {"fb_terms": 10, "fb_docs": 10, "original_query_weight": 0.5}
