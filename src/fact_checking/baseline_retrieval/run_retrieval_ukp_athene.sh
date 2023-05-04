WIKI_PAGES=".index/FEVER_wikipages/wiki-pages"

INFILE=".data/FEVER/shared_task_dev.jsonl"
OUTFILE="retrieved_data/fever_ukp/run-shared_dev-ukp-athene.jsonl"
OUTFILE_TSV="retrieved_data/fever_ukp/run-shared_dev-ukp-athene.tsv"

python flatqa/fact_checking/baseline_retrieval/list5/ukp-athene/doc_retrieval.py \
  --db-file $WIKI_PAGES --in-file $INFILE --out-file $OUTFILE || exit 1

python flatqa/fact_checking/baseline_retrieval/list5/ukp-athene/convert_to_run.py \
  --dataset_file $OUTFILE --output_run_file $OUTFILE_TSV

INFILE=".data/FEVER/baseline_data/collections/fever/train.jsonl"
OUTFILE="retrieved_data/fever_ukp/run-train-ukp-athene.jsonl"
OUTFILE_TSV="retrieved_data/fever_ukp/run-train-ukp-athene.tsv"

python flatqa/fact_checking/baseline_retrieval/list5/ukp-athene/doc_retrieval.py \
  --db-file $WIKI_PAGES --in-file $INFILE --out-file $OUTFILE || exit 1

python flatqa/fact_checking/baseline_retrieval/list5/ukp-athene/convert_to_run.py \
  --dataset_file $OUTFILE --output_run_file $OUTFILE_TSV

INFILE=".data/FEVER/shared_task_test.jsonl"
OUTFILE="retrieved_data/fever_ukp/run-shared_test-ukp-athene.jsonl"
OUTFILE_TSV="retrieved_data/fever_ukp/run-shared_test-ukp-athene.tsv"

python flatqa/fact_checking/baseline_retrieval/list5/ukp-athene/doc_retrieval.py \
  --db-file $WIKI_PAGES --in-file $INFILE --out-file $OUTFILE || exit 1

python flatqa/fact_checking/baseline_retrieval/list5/ukp-athene/convert_to_run.py \
  --dataset_file $OUTFILE --output_run_file $OUTFILE_TSV
