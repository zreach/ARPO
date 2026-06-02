#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/workspace/hf/Qwen/Qwen2.5-7B-Instruct}" \
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen2.5-7B}" \
PORT="${PORT:-8006}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}" \
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}" \
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}" \
DTYPE="${DTYPE:-bfloat16}" \
bash "$(dirname "$0")/vllm_scripts/vllm_launch_qwen2.5_7b_code_only.sh"
