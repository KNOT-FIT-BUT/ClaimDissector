python flatqa/fact_checking/baseline_retrieval/list5/expand_docs_to_sentences.py \
 --input_run_file retrieved_data/fever_merged/run.fever-paragraph.train.tsv  \
 --collection_folder .index/FEVER_wikipages/wiki-pages \
 --output_run_file retrieved_data/fever_merged/run.fever-sentence-top-200.train.tsv \
 --k 200

python flatqa/fact_checking/baseline_retrieval/list5/expand_docs_to_sentences.py \
  --input_run_file retrieved_data/fever_merged/run.fever-paragraph.shared_task_dev.tsv \
  --collection_folder .index/FEVER_wikipages/wiki-pages \
  --output_run_file retrieved_data/fever_merged/run.fever-sentence-top-200.shared_task_dev.tsv \
  --k 200
python flatqa/fact_checking/baseline_retrieval/list5/expand_docs_to_sentences.py \
  --input_run_file retrieved_data/fever_merged/run.fever-paragraph.shared_task_test.tsv \
  --collection_folder .index/FEVER_wikipages/wiki-pages \
  --output_run_file retrieved_data/fever_merged/run.fever-sentence-top-200.shared_task_test.tsv \
  --k 200
