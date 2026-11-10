# bash train_qwen_flag_test.sh
set -e

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 \
  -m FlagEmbedding.finetune.embedder.decoder_only.base \
  --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
  --trust_remote_code True \
  --cache_dir /home/aiming/cache/model_0.6_PR_total \
  --cache_path /home/aiming/cache/data_0.6_PR_total \
  --output_dir /home/aiming/finetuned_models/qwen3-0.6b-PR-total \
  --train_data /home/aiming/flag_train_folder/flag_train_start-1900_PR_total.jsonl \
  --train_group_size 8 \
  --query_max_len 4096 \
  --passage_max_len 2048 \
  --pad_to_multiple_of 8 \
  --sentence_pooling_method last_token \
  --normalize_embeddings True \
  --temperature 0.02 \
  --query_instruction_for_retrieval 'Given a partial mathematical proof, retrieve the statement it invokes.' \
  --query_instruction_format 'Instruct: {}\nQuery:{}' \
  --knowledge_distillation False \
  --learning_rate 1e-5 \
  --bf16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --negatives_cross_device \
  --gradient_checkpointing \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --dataloader_drop_last True \
  --warmup_ratio 0.1 \
  --logging_steps 10 \
  --save_steps 10000 \
  --use_lora False \
  --report_to none
