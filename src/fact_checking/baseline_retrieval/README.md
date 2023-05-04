Firstly, run
```shell
src/fact_checking/baseline_retrieval/gen_queries_qrels.sh
```

Run BM25 retrieval

```shell
src/fact_checking/baseline_retrieval/run_retrieval_bm25.sh
```

Then run UKP Athene's retrieval (yes this takes a while...)

```shell
src/fact_checking/baseline_retrieval/run_retrieval_ukp_athene.sh 
```

Now merge retrieval runs

```shell
src/fact_checking/baseline_retrieval/merge_retrieval_runs.sh
```

Expand documents to document+sentence_ids

```shell
src/fact_checking/baseline_retrieval/expand_docs_to_sentences.sh
```

(Optional step) - Generate training data
```shell
src/fact_checking/baseline_retrieval/generate_reranking_dataset.sh
```

Final step - convert run result to result accepted by this work's code
```shell
python -m src.fact_checking.baseline_retrieval.convert_runresult_to_myresult
```
