python flatqa/fact_checking/baseline_retrieval/list5/merge_runs.py \
  --input_run_file retrieved_data/fever/run.fever-anserini-paragraph.train.tsv \
  --input_run_file retrieved_data/fever_ukp/run-train-ukp-athene.tsv \
  --output_run_file retrieved_data/fever_merged/run.fever-paragraph.train.tsv \
  --strategy zip

python flatqa/fact_checking/baseline_retrieval/list5/merge_runs.py \
  --input_run_file retrieved_data/fever/run.fever-anserini-paragraph.shared_task_dev.tsv \
  --input_run_file retrieved_data/fever_ukp/run-shared_dev-ukp-athene.tsv \
  --output_run_file retrieved_data/fever_merged/run.fever-paragraph.shared_task_dev.tsv \
  --strategy zip

python flatqa/fact_checking/baseline_retrieval/list5/merge_runs.py \
  --input_run_file retrieved_data/fever/run.fever-anserini-paragraph.shared_task_test.tsv \
  --input_run_file retrieved_data/fever_ukp/run-shared_test-ukp-athene.tsv \
  --output_run_file retrieved_data/fever_merged/run.fever-paragraph.shared_task_test.tsv \
  --strategy zip
