#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${EVAL_DIR}"

mkdir -p logs

MODEL_PATH="${MODEL_PATH:-/workspace/hf/Qwen/QwQ-32B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-QwQ-32B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"

export CUDA_VISIBLE_DEVICES

echo "Starting ${SERVED_MODEL_NAME} on ${HOST}:${PORT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

exec vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
