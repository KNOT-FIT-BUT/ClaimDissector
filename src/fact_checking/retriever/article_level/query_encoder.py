# Written by Martin Fajcik <martin.fajcik@vut.cz>
#
import logging

from jsonlines import jsonlines
from pyserini.dsearch import SimpleDenseSearcher
from pyserini.search import SimpleSearcher
from tqdm import tqdm

from ....common.db import PassageDB
from ....common.utility import mkdir, count_lines


def _configure_parameters(searcher, config):
    searcher.set_bm25(**config["BM25_parameters"])
    if config["use_RM3"]:
        searcher.set_rm3(**config["RM3_parameters"])
    else:
        searcher.unset_rm3()


class QueryEncoderFramework:
    @staticmethod
    def extract_retrieval(config, input_file, output_file, eval_only=False):
        db = PassageDB(db_path=config['db_path'])
        if config["search_type"] == "DPR":
            searcher = SimpleDenseSearcher(config['index_path'],
                                           query_encoder='facebook/dpr-question_encoder-multiset-base')
        elif config["search_type"] == "BM25":
            searcher = SimpleSearcher(config['index_path'])
            _configure_parameters(searcher, config)

            logger.info(f"RM3 active: {searcher.is_using_rm3()}")
        else:
            raise ValueError(f"Unknown search type: {config['search_type']}")
        if not eval_only:
            mkdir(config["output_directory"])
        QueryEncoderFramework._extract(searcher, input_file=input_file,
                                       output_file=output_file,
                                       eval_only=eval_only,
                                       threads=config["threads"], K_extract=config["topK_extract"],
                                       batch_size=config["batch_size"], K_eval=config["topK_eval"],
                                       db=db)

    @staticmethod
    def _extract(searcher, input_file, output_file, threads, K_extract, K_eval, batch_size, db, eval_only):
        if K_extract < K_eval:
            raise ValueError("K_extract<K_eval: Cannot evaluate on more documents that are extracted!")
        with jsonlines.open(input_file) as reader:
            if not eval_only:
                writer = jsonlines.open(output_file, "w")
            buffer = []
            total_hits, total_processed = 0, 0
            total_samples = count_lines(input_file)
            pbar = tqdm(reader, total=total_samples)
            for e in reader:
                assert False, "This code is broken, check out query encoder for blocks for details"
                if len(buffer) == batch_size:
                    # extract hits
                    hits = QueryEncoderFramework._run_retrieval(K_extract, buffer, searcher, threads,
                                                                key=lambda x: x["claim"])

                    # process hit objects into simple format of List[List[doc_ids]]
                    processed_hits = [[int(hit.docid) for hit in hits[str(i)]] for i in range(len(buffer))]
                    processed_hits_scores = [[hit.score for hit in hits[str(i)]] for i in range(len(buffer))]
                    # truncate inner lists for evaluation purposes (computing accuracy@K)
                    truncated_process_hits = [x[:K_eval] for x in processed_hits]

                    gt_doc_ids = [
                        [list(set([evidence_item[2] for evidence_item in evidence_group_from_single_annotator]))
                         for evidence_group_from_single_annotator in e['evidence']] for e
                        in buffer if e["label"] != "NOT ENOUGH INFO"]

                    db_doc_ids = [
                        [db.get_doc_text(h, table="documents", columns="document_title")[0] for h in e_hits] for
                        e_hits in truncated_process_hits]

                    filtered_doc_ids = [ids for ids, e in zip(db_doc_ids, buffer) if e["label"] != "NOT ENOUGH INFO"]
                    # filtered_doc_ids = [[[""]]*len(gt_doc_ids)]

                    # Compute hits for SUPPORTED / REFUTED samples
                    # code inspired with original script's evidence retrieval https://github.com/sheffieldnlp/fever-scorer/blob/4801615100fbf6327f8e99b5dbaefe5dd890e869/src/fever/scorer.py#L37
                    # if 1 annotated group is hit, prediction is counted as hit
                    topk_gt_indices_batch = []
                    for topk_pred_ids, gt_id_groups in zip(filtered_doc_ids, gt_doc_ids):
                        total_processed += 1
                        topk_pred_ids_s = set(topk_pred_ids)
                        topk_gt_indices = []
                        for annotated_group in gt_id_groups:
                            if all([doc_id in topk_pred_ids_s for doc_id in annotated_group]):
                                total_hits += 1
                                topk_gt_indices = [topk_pred_ids.index(doc_id) for doc_id in annotated_group]
                                break

                        topk_gt_indices_batch.append(topk_gt_indices)

                    # update progress bar
                    pbar.update(n=len(buffer))
                    pbar.set_description(f"Recall@{K_eval}: {(total_hits / total_processed) * 100.:.3f}")

                    assert len(buffer) == len(processed_hits)

                    if not eval_only:
                        # write-out data
                        i = 0
                        for example, hits, scores in zip(buffer, processed_hits, processed_hits_scores):
                            evidence_hits = None
                            if example["label"] != "NOT ENOUGH INFO":
                                evidence_hits = topk_gt_indices_batch[i]
                                i += 1
                            example.update({
                                "hit_ranks": evidence_hits,
                                "predicted_indices": hits,
                                "predicted_scores": scores,
                            })
                            writer.write(example)

                    # clean buffer
                    buffer = []
                else:
                    buffer.append(e)
            if not eval_only:
                writer.close()
            logger.info(f"Final accuracy@{K_eval}: {(total_hits / total_processed) * 100.:.3f}")

    @staticmethod
    def _run_retrieval(K_extract, buffer, searcher, threads, key):
        queries = [key(example) for example in buffer]
        qids = [f"{i}" for i in range(len(queries))]
        if type(searcher) == SimpleSearcher:
            # workaround, batch_search is not thread-safe with RM3, see https://github.com/castorini/pyserini/issues/831
            if searcher.is_using_rm3():
                hits = {str(i): searcher.search(q, k=K_extract) for i, q in enumerate(queries)}
            else:
                hits = searcher.batch_search(queries, qids=qids, k=K_extract, threads=threads)
        elif type(searcher) == SimpleDenseSearcher:
            hits = searcher.batch_search(queries, q_ids=qids, k=K_extract, threads=threads)
        else:
            raise ValueError(f"Unsupported searcher type {type(searcher)}")
        return hits
