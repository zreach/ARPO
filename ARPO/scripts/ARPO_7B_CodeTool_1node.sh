#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARPO_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${ARPO_DIR}"

# ============================ Environment Setup ============================
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export MKL_SERVICE_FORCE_INTEL="${MKL_SERVICE_FORCE_INTEL:-1}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.8}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
export PYTHONPATH="${ARPO_DIR}/verl_arpo_entropy:${PYTHONPATH:-}"

# Use the same Python executable for launching training and for the code tool by default.
PYTHON_BIN="${PYTHON_BIN:-python}"
CODE_TOOL_PYTHON_BIN="${CODE_TOOL_PYTHON_BIN:-${PYTHON_BIN}}"

# ============================ Basic Configuration ============================
PROJECT_NAME="${PROJECT_NAME:-reasoning_code_tool}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-arpo_7b_code_tool_only}"

CONFIG_PATH="${CONFIG_PATH:-${ARPO_DIR}/scripts/config}"
CONFIG_NAME="${CONFIG_NAME:-ppo_trainer_code_tool.yaml}"

NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

# ============================ Data Configuration ============================
PROMPT_KEY="${PROMPT_KEY:-prompt}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"

SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-${ARPO_DIR}/scripts/prompts/system_prompt_code_only.txt}"
SOURCE_TRAIN_FILES="${SOURCE_TRAIN_FILES:-${ARPO_DIR}/rl_datasets/train_10k.parquet}"
SOURCE_VALID_FILES="${SOURCE_VALID_FILES:-${ARPO_DIR}/rl_datasets/valid.parquet}"
CODE_ONLY_DATA_DIR="${CODE_ONLY_DATA_DIR:-${ARPO_DIR}/rl_datasets/code_only}"
TRAIN_FILES="${TRAIN_FILES:-${CODE_ONLY_DATA_DIR}/train.parquet}"
VALID_FILES="${VALID_FILES:-${CODE_ONLY_DATA_DIR}/valid.parquet}"
REBUILD_CODE_ONLY_DATA="${REBUILD_CODE_ONLY_DATA:-False}"
CODE_ONLY_ABILITY="${CODE_ONLY_ABILITY:-math}"
CODE_ONLY_FALLBACK_VAL_SIZE="${CODE_ONLY_FALLBACK_VAL_SIZE:-256}"

# ============================ Model Configuration ============================
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-/workspace/hf/Qwen/Qwen2.5-7B-Instruct}"

# ============================ Rollout Configuration ==========================
ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
ROLLOUT_MODE="${ROLLOUT_MODE:-sync_with_tool}"
ROLLOUT_N="${ROLLOUT_N:-16}"
INITIAL_ROLLOUTS="${INITIAL_ROLLOUTS:-8}"
BEAM_SIZE="${BEAM_SIZE:-2}"
BRANCH_PROBABILITY="${BRANCH_PROBABILITY:-0.5}"
ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-0.2}"
TOOL_CALL_LIMIT="${TOOL_CALL_LIMIT:-6}"
TOOL_MAX_WORKERS="${TOOL_MAX_WORKERS:-64}"
TOOL_TIMEOUT="${TOOL_TIMEOUT:-120}"
TOOL_RETRY_COUNT="${TOOL_RETRY_COUNT:-3}"

# Optional lightweight on-policy distillation loss. Disabled by default.
USE_OPD_LOSS="${USE_OPD_LOSS:-False}"
OPD_LOSS_COEF="${OPD_LOSS_COEF:-0.0}"
OPD_POSITIVE_ONLY="${OPD_POSITIVE_ONLY:-True}"

# Keep ARPO's old multi_turn switch off. Tool execution is handled by sync_with_tool.
ENABLE_MULTI_TURN="${ENABLE_MULTI_TURN:-False}"

# ============================ Reward Configuration ==========================
REWARD_MANAGER="${REWARD_MANAGER:-naive}"
CUSTOM_REWARD_FUNCTION_PATH="${CUSTOM_REWARD_FUNCTION_PATH:-${ARPO_DIR}/verl_arpo_entropy/verl/utils/reward_score/deep_research.py}"
CUSTOM_REWARD_FUNCTION_NAME="${CUSTOM_REWARD_FUNCTION_NAME:-compute_score}"

# ============================ Training Configuration ============================
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
SAVE_FREQ="${SAVE_FREQ:-5}"
TEST_FREQ="${TEST_FREQ:-5}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[console, wandb]}"

# ============================ Path Configuration ============================
SAVE_PATH="${SAVE_PATH:-${ARPO_DIR}/checkpoints/${EXPERIMENT_NAME}}"
ROLLOUT_SAVE_PATH="${ROLLOUT_SAVE_PATH:-${SAVE_PATH}/rollout}"
mkdir -p "${SAVE_PATH}" "${ROLLOUT_SAVE_PATH}"

# Build the default code-only dataset once from ARPO's mixed train_10k data.
if [[ "${REBUILD_CODE_ONLY_DATA}" == "True" || "${REBUILD_CODE_ONLY_DATA}" == "true" || ! -f "${TRAIN_FILES}" || ! -f "${VALID_FILES}" ]]; then
    "${PYTHON_BIN}" "${ARPO_DIR}/scripts/prepare_code_only_rl_data.py" \
        --train-source "${SOURCE_TRAIN_FILES}" \
        --train-target "${TRAIN_FILES}" \
        --val-source "${SOURCE_VALID_FILES}" \
        --val-target "${VALID_FILES}" \
        --system-prompt "${SYSTEM_PROMPT_PATH}" \
        --prompt-key "${PROMPT_KEY}" \
        --ability "${CODE_ONLY_ABILITY}" \
        --fallback-val-size "${CODE_ONLY_FALLBACK_VAL_SIZE}" \
        --summary "${CODE_ONLY_DATA_DIR}/summary.json"
fi

# ============================ Start Training ============================
"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    --config-path="${CONFIG_PATH}" \
    --config-name="${CONFIG_NAME}" \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VALID_FILES}" \
    data.prompt_key="${PROMPT_KEY}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.model.path="${ACTOR_MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$((2 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)))" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    ++actor_rollout_ref.actor.use_opd_loss="${USE_OPD_LOSS}" \
    ++actor_rollout_ref.actor.opd_loss_coef="${OPD_LOSS_COEF}" \
    ++actor_rollout_ref.actor.opd_positive_only="${OPD_POSITIVE_ONLY}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$((4 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)))" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="${ROLLOUT_NAME}" \
    actor_rollout_ref.rollout.mode="${ROLLOUT_MODE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.initial_rollouts="${INITIAL_ROLLOUTS}" \
    actor_rollout_ref.rollout.beam_size="${BEAM_SIZE}" \
    actor_rollout_ref.rollout.branch_probability="${BRANCH_PROBABILITY}" \
    actor_rollout_ref.rollout.entropy_weight="${ENTROPY_WEIGHT}" \
    actor_rollout_ref.rollout.tools.call_limit="${TOOL_CALL_LIMIT}" \
    actor_rollout_ref.rollout.tools.max_workers="${TOOL_MAX_WORKERS}" \
    actor_rollout_ref.rollout.tools.timeout="${TOOL_TIMEOUT}" \
    actor_rollout_ref.rollout.tools.retry_count="${TOOL_RETRY_COUNT}" \
    ~actor_rollout_ref.rollout.tools.tool_instances.search \
    actor_rollout_ref.rollout.tools.tool_instances.python.params.python_path="${CODE_TOOL_PYTHON_BIN}" \
    actor_rollout_ref.rollout.multi_turn.enable="${ENABLE_MULTI_TURN}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$((4 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)))" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager="${REWARD_MANAGER}" \
    custom_reward_function.path="${CUSTOM_REWARD_FUNCTION_PATH}" \
    custom_reward_function.name="${CUSTOM_REWARD_FUNCTION_NAME}" \
    trainer.critic_warmup=0 \
    trainer.logger="${TRAINER_LOGGER}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.default_local_dir="${SAVE_PATH}" \
    trainer.val_before_train=False \
    trainer.rollout_data_dir="${ROLLOUT_SAVE_PATH}" \
    hydra.run.dir="${SAVE_PATH}/outputs" \
    "$@" 2>&1 | tee "${SAVE_PATH}/run.log"
