# Written by Martin Fajcik <martin.fajcik@vut.cz>
#
from jsonlines import jsonlines
from tqdm import tqdm


def convert_runresult(runfile, original_file, output_file):
    rankings = dict()
    with open(runfile, 'r', encoding='utf-8') as f:
        print(f'Reading input run file {runfile}...')
        for line in tqdm(f):
            r = line.strip().split('\t')
            if len(line.strip().split('\t')) < 3:
                query_id, _, doc_id, rank, _, _ = line.strip().split()
            else:
                query_id, doc_id, _ = r
            query_id = int(query_id)
            if query_id not in rankings:
                rankings[query_id] = {}
            if runfile not in rankings[query_id]:
                rankings[query_id][runfile] = []
            rankings[query_id][runfile].append(doc_id)

    with jsonlines.open(original_file) as reader, jsonlines.open(output_file, "w") as writer:
        data = []
        for e in reader:
            e['predicted_article_indices'] = list(rankings[e['id']].values())[0]
            data.append(e)
            writer.write(e)


if __name__ == "__main__":
    convert_runresult(runfile=".data/FEVER/fever_interleaved_retrieval/run.fever-paragraph.train.tsv",
                      original_file=".data/FEVER/baseline_data/collections/fever/train.jsonl",
                      output_file=".data/FEVER/verifier/article_level/train_ret_baseline.jsonl")
    convert_runresult(runfile=".data/FEVER/fever_interleaved_retrieval/run.fever-paragraph.shared_task_dev.tsv",
                      original_file=".data/FEVER/shared_task_dev.jsonl",
                      output_file=".data/FEVER/verifier/article_level/shared_task_dev_ret_baseline.jsonl")
    convert_runresult(runfile=".data/FEVER/fever_interleaved_retrieval/run.fever-paragraph.shared_task_test.tsv",
                      original_file=".data/FEVER/shared_task_test.jsonl",
                      output_file=".data/FEVER/verifier/article_level/shared_task_test_ret_baseline.jsonl")
