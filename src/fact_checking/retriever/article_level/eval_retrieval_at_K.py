# Written by Martin Fajcik <martin.fajcik@vut.cz>
#
import json
import unicodedata

import jsonlines
import unidecode as unidecode


def strip_accents(s):
    return unidecode.unidecode(s)


def eval_at_K(PREDS, DATASET, RETRIEVAL_INPUTS, COLLECTION):
    with open(DATASET, "r") as rf:
        official_dataset = json.load(rf)
    with open(PREDS, "r") as rf:
        retrieval_results = json.load(rf)
    with jsonlines.open(COLLECTION, "r") as reader:
        collection = {e['pid']: e for e in reader}
    total = 0
    hits = 0
    hits_with_nfd = 0
    hits_qas = 0
    hits_wo_accents = 0
    hits_soft = 0
    with jsonlines.open(RETRIEVAL_INPUTS, "r") as reader_retrieval_inputs:
        for idx, (official_sample, retrieval_input_sample) in enumerate(zip(official_dataset, reader_retrieval_inputs)):
            if official_sample["label"] != "SUPPORTED":
                continue
            total += 1
            retrieved = retrieval_results[str(retrieval_input_sample["qid"])]
            # flatten lists
            retrieved = [y for x in retrieved[1] for y in x]
            retrieved_titles = [collection[r]['title'] for r in retrieved]

            # Make sure retrieval is unique
            assert len(retrieved_titles) == len(set(retrieved_titles))

            # Following official evaluation approach
            # https://github.com/hover-nlp/hover/blob/bb475622dfaecaf505beaa28f28db11986528d80/my_transformers/data/metrics/hover_doc_metrics.py#L113
            hits += int(
                all(annotated_fact[0] in retrieved_titles for annotated_fact in official_sample['supporting_facts']))

            # with NFD normalization
            nfd_retrieved_titles = [unicodedata.normalize('NFD', t) for t in retrieved_titles]
            hits_with_nfd += int(
                all(unicodedata.normalize('NFD', annotated_fact[0]) in nfd_retrieved_titles for annotated_fact in
                    official_sample['supporting_facts']))

            # mapping to ASCII
            ascii_retrieved_titles = [strip_accents(t) for t in retrieved_titles]
            hits_wo_accents += int(
                all(strip_accents(annotated_fact[0]) in ascii_retrieved_titles for annotated_fact in
                    official_sample['supporting_facts']))

            # Try based on GT data in RETRIEVAL_INPUTS
            hits_qas += int(all(pid in retrieved for pid in retrieval_input_sample['support_pids']))

            hits_soft += len(set.intersection(set(retrieval_input_sample['support_pids']), set(retrieved))) / \
                         max(1.0, len(set(retrieval_input_sample['support_pids'])))

    print(f"RETRIEVAL: {hits / total * 100.:.4f}")
    print(f"RETRIEVAL_NFD: {hits_with_nfd / total * 100.:.4f}")
    print(f"RETRIEVAL_ASCII: {hits_wo_accents / total * 100.:.4f}")
    print(f"RETRIEVAL_PREPQAS: {hits_qas / total * 100.:.4f}")

    # BUT THIS IS NOT WHAT IS IN THE PAPER :-(
    # FROM COLBERT
    # https://github.com/stanford-futuredata/ColBERT/blob/4120febb8a1cb8018f4a64edec7928d2d01ff503/colbert/evaluation/metrics.py#L102
    """
    The first is Retrieval@100, which is the
    percentage of claims for which the system retrieves all of the relevant passages within the top-100
    results
    """
    print(f"MACRO_RECALL: {hits_soft / total * 100.:.4f}")


if __name__ == "__main__":
    PREDS = ".data/HOVER/verifier/article_level/baleen_inference_25x4_ordered/output_separate_deduplists_dev.json"
    RETRIEVAL_INPUTS = ".data/HOVER/baleen_retrieval/dev/qas.jsonl"
    QUESTION_RET_INPUTS = ".data/HOVER/baleen_retrieval/train/questions.tsv"
    DATASET = ".data/HOVER/official/hover_dev_release_v1.1.json"
    COLLECTION = ".index/HoVer/collection.json"
    eval_at_K(PREDS, DATASET, RETRIEVAL_INPUTS, COLLECTION)
