#bash eval_qwen_flag.sh
set -euo pipefail

python -m FlagEmbedding.evaluation.custom \
    --eval_name CO_qwen3_eval_ALL \
    --dataset_dir ./datadictALL \
    --splits test \
    --corpus_embd_save_dir ./datadictALL/corpus_save_ALL_qwen_eval_CO \
    --output_dir ./datadictALL/search_results_ALL_qwen_eval_CO \
    --search_top_k 1000 \
    --eval_output_method markdown \
    --eval_output_path ./datadictALL/eval_results_ALL_qwen_eval_CO.md \
    --eval_metrics ndcg_at_10 recall_at_100 \
    --embedder_name_or_path /home/h_proctor/output/qwen3-0.6b-math-CO_20k \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 \
    --embedder_batch_size 32 \
    --embedder_model_class decoder-only-base
