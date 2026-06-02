#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${EVAL_DIR}"

mkdir -p logs

MODEL_PATH="${MODEL_PATH:-/workspace/hf/Qwen/Qwen2.5-7B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen2.5-7B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8006}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
DTYPE="${DTYPE:-bfloat16}"
API_KEY="${API_KEY:-EMPTY}"
SERVED_LOG="${SERVED_LOG:-logs/qwen2.5_7b_code_only_vllm.log}"

export CUDA_VISIBLE_DEVICES

echo "Starting ${SERVED_MODEL_NAME} on ${HOST}:${PORT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "DTYPE=${DTYPE}"

exec vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --dtype "${DTYPE}" \
    --api-key "${API_KEY}" \
    2>&1 | tee "${SERVED_LOG}"
