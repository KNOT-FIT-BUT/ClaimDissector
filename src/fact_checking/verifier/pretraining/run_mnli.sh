# deberta base was trained with the same hyperparameters

export OUTPUT_DIR="transformers/.pretraining_output"

python -m torch.distributed.launch \
  --nproc_per_node 2 run_glue.py \
  --model_name_or_path microsoft/deberta-v3-large --task_name mnli \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir $OUTPUT_DIR \
  --fp16
