# Written by Martin Fajcik <martin.fajcik@vut.cz>
#
import collections
import json
import sys

from .official.hover_sent_metrics import hover_evaluate as hover_sent_evaluate, normalize_sp
from .official.hover_verif_metrics import hover_evaluate as hover_verif_evaluate


def hover_score_eval_ours(examples, preds):
    """
    Explanation of HoVer score from the official website https://hover-nlp.github.io/

    As explained in the Sec 5.5 of the paper, the HoVer score is the percentage
    of examples where the model must retrieve at least one supporting fact from
    every supporting document and predict the correct label. However, this doesn't
    mean submitting a large number of supporting facts can boost up the score.
    When calculating the HoVer score for a k-hop example, we are only gonna consider
    supporting facts (sentences) from the first k+1 documents retrieved by your model.
    For each document considered, we are only gonna evaluate the top-2 selected sentences.
    So remember to rank your retrieved supporting facts! Please refer to our evaluation
    script provided below for calculating the supporting-fact F1 score and HoVer score (Coming Soon).
    """
    total = 0
    hits = 0
    for example in examples:
        total += 1
        uid = example["uid"]
        gold_label = example['label']
        supporting_facts = example["supporting_facts"]
        K = example["num_hops"]

        supporting_documents = collections.defaultdict(lambda: [])
        for title, sentid in supporting_facts:
            supporting_documents[title].append(sentid)

        if uid not in preds:
            print("Missing prediction for %s" % uid)
            continue

        prediction = preds[uid]
        pred_label = preds[uid]['predicted_label']
        predicted_ranking = prediction['top_predicted_evidence']

        # get titles and top-2 sentence predictions of K+1 documents
        filtered_predictions = collections.defaultdict(lambda: [])
        for p in predicted_ranking:
            if len(filtered_predictions) > K + 1:
                break
            if len(filtered_predictions[p[0]]) < 2:
                filtered_predictions[p[0]].append(p[1])

        def has_facts(supporting_documents, filtered_predictions):
            # get titles of all correct documents
            for supporting_document in supporting_documents.keys():
                # model must retrieve at least one supporting fact from
                # every supporting document and predict the correct label
                if supporting_document in filtered_predictions:
                    predicted_sentences = filtered_predictions[supporting_document]
                    assert len(predicted_sentences) <= 2
                    sentence_hit = any(sentid in supporting_documents[supporting_document] for sentid in
                                       filtered_predictions[supporting_document])
                    if not sentence_hit:
                        return False
                else:
                    return False
            return True

        fact_hit = has_facts(supporting_documents, filtered_predictions)
        class_hit = int(pred_label == gold_label)
        hits += int(fact_hit and class_hit)
    return {"hover_score": hits / total}


def hover_eval(model_predictions, return_dict=False):
    "Wrapper for passing model predictions to hover_evaluation"
    examples = [e['example'] for e in model_predictions]
    predictions = {e['example']['uid']: e for e in model_predictions}
    try:
        r = hover_evaluation(examples, predictions)
    except ZeroDivisionError:
        return -1, -1, -1, -1
    if return_dict:
        return r
    return r['hover_score'], r['acc'], r['exact'], r['f1']


def hover_evaluation(examples, predictions):
    results = hover_sent_evaluate(examples, predictions)
    results.update(hover_verif_evaluate(examples, predictions))

    results.update(hover_score_eval_ours(examples, predictions))
    return results


if __name__ == "__main__":
    import pickle

    with open("predictions_hover.pkl", "rb") as f:
        preds = pickle.load(f)
    result = hover_eval(model_predictions=preds)

    # PRED_FILE = sys.argv[1]
    # REFERENCE_SAMPLE_FILE = sys.argv[2]
    # with open(PRED_FILE) as pf, open(REFERENCE_SAMPLE_FILE) as rf:
    #     predictions = json.load(pf)
    #     references = json.load(rf)
    # result = hover_evaluation(references, predictions)
    print(json.dumps(result, indent=4))
