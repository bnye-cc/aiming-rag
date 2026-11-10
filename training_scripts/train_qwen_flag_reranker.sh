set -e

export WANDB_MODE=disabled

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


MODEL=Qwen/Qwen3-Reranker-0.6B
TRAIN_DATA=/home/h_proctor/output/qwen3-4b-math-NT_50k
CACHE_DIR=/home/h_proctor/cache/model_reranker_NT
CACHE_PATH=/home/h_proctor/cache/data_reranker_NT
OUTPUT_DIR=/home/h_proctor/output/qwen3-reranker-0.6b-NT


QUERY_MAX_LEN=4096
PASSAGE_MAX_LEN=2048

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 \
  -m FlagEmbedding.finetune.reranker.decoder_only.base \
  --model_name_or_path $MODEL \
  --model_type decoder \
  --trust_remote_code True \
  --cache_dir $CACHE_DIR \
  --cache_path $CACHE_PATH \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir \
  --train_data $TRAIN_DATA \
  --train_group_size 8 \
  --query_max_len $QUERY_MAX_LEN \
  --passage_max_len $PASSAGE_MAX_LEN \
  --pad_to_multiple_of 8 \
  --knowledge_distillation False \
  --query_instruction_for_rerank 'Given a partial mathematical proof, retrieve the statement it invokes.' \
  --query_instruction_format '{}\n{}' \
  --passage_instruction_for_rerank '' \
  --passage_instruction_format '{}{}' \
  --use_lora False \
  --use_flash_attn False \
  --learning_rate 1e-5 \
  --bf16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --dataloader_drop_last True \
  --warmup_ratio 0.1 \
  --logging_steps 10 \
  --save_steps 10000 \
  --report_to none
