#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Base OPD from "KL for a KL": on-policy sampled-token distillation with
# r_t = log pi_T(y_t|c_t) - log pi_theta(y_t|c_t), without vOPD's baseline.
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-/workspace/hf/Qwen/Qwen2.5-3B-Instruct}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/workspace/hf/Qwen/Qwen2.5-7B-Instruct}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-base_opd_3b_code_tool_only}"

# Keep the earlier lightweight OPD branch disabled; this script uses paper-style base OPD.
export USE_OPD_LOSS="${USE_OPD_LOSS:-False}"
export OPD_LOSS_COEF="${OPD_LOSS_COEF:-0.0}"

BASE_OPD_LOSS_COEF="${BASE_OPD_LOSS_COEF:-1.0}"
BASE_OPD_USE_TASK_REWARDS="${BASE_OPD_USE_TASK_REWARDS:-False}"
BASE_OPD_USE_POLICY_GRADIENT="${BASE_OPD_USE_POLICY_GRADIENT:-True}"
BASE_OPD_REWARD_MAX_CLAMP="${BASE_OPD_REWARD_MAX_CLAMP:-10.0}"

exec bash "${SCRIPT_DIR}/ARPO_3B_CodeTool_1node.sh" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    ++actor_rollout_ref.ref.model_path="${TEACHER_MODEL_PATH}" \
    ++actor_rollout_ref.actor.use_base_opd_loss=True \
    ++actor_rollout_ref.actor.base_opd_loss_coef="${BASE_OPD_LOSS_COEF}" \
    ++actor_rollout_ref.actor.base_opd_use_task_rewards="${BASE_OPD_USE_TASK_REWARDS}" \
    ++actor_rollout_ref.actor.base_opd_use_policy_gradient="${BASE_OPD_USE_POLICY_GRADIENT}" \
    ++actor_rollout_ref.actor.base_opd_reward_max_clamp="${BASE_OPD_REWARD_MAX_CLAMP}" \
    "$@"
