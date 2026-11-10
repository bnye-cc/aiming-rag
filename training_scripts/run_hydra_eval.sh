#!/bin/bash
# run_hydra.sh -- routed multi-embedder eval on one FlagEmbedding eval set.
#
# Usage:
#   bash run_hydra.sh BALALL          # runs on ./datadictBALALL
#   bash run_hydra.sh PR              # runs on ./datadictPR
#   bash run_hydra.sh ALL
#
#   # all 8 sets, one after another:
#   for s in BALALL ALL CO AG NT GT FA PR; do bash run_hydra.sh "$s"; done
#
# Resume behavior: routing and finished buckets are reused automatically on
# rerun. To force redo:  FORCE=1 bash run_hydra.sh BALALL

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: bash run_hydra.sh <EVALSET>   (e.g. BALALL, ALL, CO, AG, NT, GT, FA, PR)"
    exit 1
fi
EVALSET="$1"

# --------------------------------------------------------------------------
# paths -- edit here if anything moves
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${SCRIPT_DIR}/datadict${EVALSET}"
WORKDIR="${SCRIPT_DIR}/hydra_runs_4b/${EVALSET}"

CLASSIFIER="/home/aiming/category_classifier/qwen35_4b_category6_lora_v1/best_model"
CLASSIFY_PY="${SCRIPT_DIR}/classify.py"

EMB_ROOT="/home/aiming/finetuned_models"
EMB_CO="${EMB_ROOT}/qwen3-4b-math-CO_50k"
EMB_AG="${EMB_ROOT}/qwen3-4b-math-AG_50k"
EMB_NT="${EMB_ROOT}/qwen3-4b-math-NT_50k"
EMB_GT="${EMB_ROOT}/qwen3-4b-math-GT_50k"
EMB_FA="${EMB_ROOT}/qwen3-4b-math-FA_50k"
EMB_PR="${EMB_ROOT}/qwen3-4b-math-PR_50k"

DEVICES="cuda:0 cuda:1 cuda:2 cuda:3"
SEARCH_TOP_K=1000
EMBEDDER_BATCH_SIZE=8
EMBEDDER_MODEL_CLASS="decoder-only-base"
EVAL_METRICS="ndcg_at_10 recall_at_100"

# --------------------------------------------------------------------------
# environment -- classifier JITs the fla Triton kernels, so gcc must be loaded
# --------------------------------------------------------------------------
module load gcc/15.2 python/3.11.14
export CC="$(which gcc)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /home/aiming/qwen_env_311/bin/activate

# --------------------------------------------------------------------------
# sanity checks before burning GPU time
# --------------------------------------------------------------------------
for f in test_queries.jsonl test_qrels.jsonl corpus.jsonl; do
    [ -f "${DATASET_DIR}/${f}" ] || { echo "ERROR: missing ${DATASET_DIR}/${f}"; exit 1; }
done
[ -d "${CLASSIFIER}" ] || { echo "ERROR: classifier not found: ${CLASSIFIER}"; exit 1; }
for p in "$EMB_CO" "$EMB_AG" "$EMB_NT" "$EMB_GT" "$EMB_FA" "$EMB_PR"; do
    [ -d "$p" ] || { echo "ERROR: embedder not found: $p"; exit 1; }
done

FORCE_FLAGS=""
if [ "${FORCE:-0}" = "1" ]; then
    FORCE_FLAGS="--force_search"
else
    # reuse routing.jsonl if a previous run produced it
    [ -f "${WORKDIR}/routing.jsonl" ] && FORCE_FLAGS="--skip_routing"
fi

echo "=== hydra eval: ${EVALSET} ==="
echo "  dataset   : ${DATASET_DIR}"
echo "  workdir   : ${WORKDIR}"
echo "  classifier: ${CLASSIFIER}"
echo "  flags     : ${FORCE_FLAGS:-none}"

python "${SCRIPT_DIR}/hydra_eval.py" \
    --dataset_dir "${DATASET_DIR}" \
    --workdir "${WORKDIR}" \
    --run_name "${EVALSET}" \
    --classifier "${CLASSIFIER}" \
    --classify_py "${CLASSIFY_PY}" \
    --embedder_CO "${EMB_CO}" \
    --embedder_AG "${EMB_AG}" \
    --embedder_NT "${EMB_NT}" \
    --embedder_GT "${EMB_GT}" \
    --embedder_FA "${EMB_FA}" \
    --embedder_PR "${EMB_PR}" \
    --devices ${DEVICES} \
    --search_top_k "${SEARCH_TOP_K}" \
    --embedder_batch_size "${EMBEDDER_BATCH_SIZE}" \
    --embedder_model_class "${EMBEDDER_MODEL_CLASS}" \
    --eval_metrics ${EVAL_METRICS} \
    ${FORCE_FLAGS}

echo ""
echo "=== results: ${WORKDIR}/hydra_results.md ==="
cat "${WORKDIR}/hydra_results.md"
