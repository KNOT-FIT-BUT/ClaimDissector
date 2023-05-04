python flatqa/fact_checking/baseline_retrieval/generate_queries_and_qrels.py \
  --dataset_file .data/FEVER/baseline_data/collections/fever/train.jsonl \
  --output_queries_file .data/FEVER/baseline_data/processed/queries.paragraph.train.tsv \
  --output_qrels_file .data/FEVER/baseline_data/processed/qrels.paragraph.train.txt \
  --granularity paragraph

python flatqa/fact_checking/baseline_retrieval/generate_queries_and_qrels.py \
  --dataset_file .data/FEVER/shared_task_dev.jsonl --output_queries_file .data/FEVER/baseline_data/processed/queries.paragraph.shared_task_dev.tsv \
  --output_qrels_file .data/FEVER/baseline_data/processed/qrels.paragraph.shared_task_dev.txt \
  --granularity paragraph

python flatqa/fact_checking/baseline_retrieval/generate_queries_and_qrels.py \
  --dataset_file .data/FEVER/shared_task_test.jsonl --output_queries_file .data/FEVER/baseline_data/processed/queries.paragraph.shared_task_test.tsv \
  --output_qrels_file .data/FEVER/baseline_data/processed/qrels.paragraph.shared_task_test.txt \
  --granularity paragraph
