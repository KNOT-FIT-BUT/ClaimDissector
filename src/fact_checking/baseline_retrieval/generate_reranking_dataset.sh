python flatqa/fact_checking/baseline_retrieval/list5/generate_sentence_selection_data.py \
  --dataset_file .data/FEVER/baseline_data/collections/fever/train.jsonl \
  --run_file retrieved_data/fever_merged/run.fever-sentence-top-200.train.tsv \
  --collection_folder .index/FEVER_wikipages/wiki-pages \
  --output_id_file retrieved_data/fever_merged/prepared_data_from_run.fever-sentence-top-200.train_ids.tsv \
  --output_text_file retrieved_data/fever_merged/prepared_data_from_run.fever-sentence-top-200.train_texts.tsv \
  --min_rank 50 \
  --max_rank 200 \
  --seed 1234
