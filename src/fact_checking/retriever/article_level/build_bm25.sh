INPUT_FOLDER_WITH_JSONL_FILES=".index/FEVER_wikipages/prep_anserini/"
OUTPUT_FOLDER_WITH_INDEX=".index/feverwiki_bm25_index_anserini"
THREADS=48

python -m pyserini.index -collection JsonCollection \
  -generator DefaultLuceneDocumentGenerator \
  -threads "$THREADS" \
  -input "$INPUT_FOLDER_WITH_JSONL_FILES" \
  -index "$OUTPUT_FOLDER_WITH_INDEX" \
  -storePositions -storeDocvectors -storeRaw
