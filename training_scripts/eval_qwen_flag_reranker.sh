#bash eval_qwen_flag_reranker.sh
set -euo pipefail

python -m FlagEmbedding.evaluation.custom \
    --eval_name base_qwen3_eval2_qwen3rr \
    --dataset_dir ./datadictALL \
    --splits test \
    --corpus_embd_save_dir ./datadict/corpus_save_100k_qwen_eval2_qwen3rr \
    --output_dir ./datadict/search_results_100k_qwen_eval2_qwen3rr \
    --search_top_k 1000 \
    --eval_output_method markdown \
    --eval_output_path ./datadictALL/eval_results_100k_qwen_base_qwen3rr.md \
    --eval_metrics ndcg_at_10 recall_at_100 \
    --embedder_name_or_path /home/h_proctor/qwen3-0.6b-math-clean-100k_v2 \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 \
    --embedder_batch_size 32 \
    --embedder_query_max_length 4096 \
    --embedder_passage_max_length 2048 \
    --rerank_top_k 50 \
    --reranker_name_or_path Qwen/Qwen3-Reranker-0.6B \
    --reranker_model_class decoder-only-base \
    --reranker_query_max_length 2048 \
    --reranker_max_length 3072 \
    --reranker_batch_size 16 \
    --embedder_model_class decoder-only-base
