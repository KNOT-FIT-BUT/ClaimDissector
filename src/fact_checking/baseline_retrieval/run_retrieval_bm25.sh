INDEX=".index/feverwiki_bm25_jianetal"

#
export JAVA_HOME="/usr/lib/jvm/jdk-16/"
ANSERINI_FOLDER="$HOME/anserini"
THREADS=4


TOPICS=".data/FEVER/baseline_data/processed/queries.paragraph.train.tsv"
OUTPUT="retrieved_data/fever/run.fever-anserini-paragraph.train.tsv"
sh $ANSERINI_FOLDER/target/appassembler/bin/SearchCollection \
  -index $INDEX \
  -topicreader TsvInt -topics $TOPICS \
  -output $OUTPUT \
  -threads $THREADS \
  -bm25 -bm25.k1 0.6 -bm25.b 0.5

TOPICS=".data/FEVER/baseline_data/processed/queries.paragraph.shared_task_dev.tsv"
OUTPUT="retrieved_data/fever/run.fever-anserini-paragraph.shared_task_dev.tsv"
sh $ANSERINI_FOLDER/target/appassembler/bin/SearchCollection \
  -index $INDEX \
  -topicreader TsvInt -topics $TOPICS \
  -output $OUTPUT \
  -threads $THREADS \
  -bm25 -bm25.k1 0.6 -bm25.b 0.5

TOPICS=".data/FEVER/baseline_data/processed/queries.paragraph.shared_task_test.tsv"
OUTPUT="retrieved_data/fever/run.fever-anserini-paragraph.shared_task_test.tsv"
sh $ANSERINI_FOLDER/target/appassembler/bin/SearchCollection \
  -index $INDEX \
  -topicreader TsvInt -topics $TOPICS \
  -output $OUTPUT \
  -threads $THREADS \
  -bm25 -bm25.k1 0.6 -bm25.b 0.5
