#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARPO_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${ARPO_DIR}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VERL_SFT_LOGGING_LEVEL="${VERL_SFT_LOGGING_LEVEL:-INFO}"
export PYTHONPATH="${ARPO_DIR}/verl_arpo_entropy:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

MODEL_PATH="${MODEL_PATH:-/workspace/hf/Qwen/Qwen2.5-7B-Instruct}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"

PROJECT_NAME="${PROJECT_NAME:-code_only_sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2.5_7b_code_only_sft}"
SAVE_PATH="${SAVE_PATH:-${ARPO_DIR}/checkpoints/${EXPERIMENT_NAME}}"

DATA_JSONL="${DATA_JSONL:-${ARPO_DIR}/../../my-verl/datasets/dongguanting/Tool-Star-SFT-54K-code-tool-math-only/data.jsonl}"
DATA_DIR="${DATA_DIR:-${ARPO_DIR}/rl_datasets/code_only_sft}"
TRAIN_PARQUET="${TRAIN_PARQUET:-${DATA_DIR}/train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${DATA_DIR}/val.parquet}"
VAL_SIZE="${VAL_SIZE:-512}"
SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-${ARPO_DIR}/scripts/prompts/system_prompt_code_only.txt}"
REBUILD_SFT_DATA="${REBUILD_SFT_DATA:-True}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
TRUNCATION="${TRUNCATION:-right}"
LR="${LR:-1e-5}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:--1}"
TEST_FREQ="${TEST_FREQ:--1}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[console]}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"

mkdir -p "${SAVE_PATH}" "${DATA_DIR}"

if [[ "${REBUILD_SFT_DATA}" == "True" || "${REBUILD_SFT_DATA}" == "true" || ! -f "${TRAIN_PARQUET}" || ! -f "${VAL_PARQUET}" ]]; then
    "${PYTHON_BIN}" "${ARPO_DIR}/scripts/prepare_code_only_sft_data.py" \
        --source "${DATA_JSONL}" \
        --train "${TRAIN_PARQUET}" \
        --val "${VAL_PARQUET}" \
        --system-prompt "${SYSTEM_PROMPT_PATH}" \
        --val-size "${VAL_SIZE}" \
        --summary "${DATA_DIR}/summary.json"
fi

"${TORCHRUN_BIN}" --standalone --nnodes="${NNODES}" --nproc_per_node="${NPROC_PER_NODE}" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="${VAL_PARQUET}" \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    data.max_length="${MAX_LENGTH}" \
    data.truncation="${TRUNCATION}" \
    model.partial_pretrain="${MODEL_PATH}" \
    model.fsdp_config.model_dtype="${MODEL_DTYPE}" \
    model.enable_gradient_checkpointing=true \
    optim.lr="${LR}" \
    trainer.default_local_dir="${SAVE_PATH}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger="${TRAINER_LOGGER}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
    use_remove_padding=true \
    "$@" 2>&1 | tee "${SAVE_PATH}/run.log"
