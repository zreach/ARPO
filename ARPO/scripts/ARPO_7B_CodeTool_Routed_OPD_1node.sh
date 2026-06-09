#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-/workspace/hf/Qwen/Qwen2.5-7B-Instruct}"
MATH_TEACHER_MODEL_PATH="${MATH_TEACHER_MODEL_PATH:-/workspace/hf/Qwen/Qwen2.5-Math-7B-Instruct}"
CODE_TEACHER_MODEL_PATH="${CODE_TEACHER_MODEL_PATH:-/workspace/hf/Qwen/Qwen2.5-Coder-7B-Instruct}"

export ACTOR_MODEL_PATH
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-routed_opd_7b_code_tool_only}"

# Keep the earlier lightweight OPD branch disabled; this script uses routed OPD.
export USE_OPD_LOSS="${USE_OPD_LOSS:-False}"
export OPD_LOSS_COEF="${OPD_LOSS_COEF:-0.0}"

ROUTED_OPD_LOSS_MODE="${ROUTED_OPD_LOSS_MODE:-k1}"
ROUTED_OPD_LOSS_COEF="${ROUTED_OPD_LOSS_COEF:-1.0}"
ROUTED_OPD_USE_TASK_REWARDS="${ROUTED_OPD_USE_TASK_REWARDS:-True}"
ROUTED_OPD_USE_POLICY_GRADIENT="${ROUTED_OPD_USE_POLICY_GRADIENT:-True}"
ROUTED_OPD_LOSS_MAX_CLAMP="${ROUTED_OPD_LOSS_MAX_CLAMP:-10.0}"

exec "${SCRIPT_DIR}/ARPO_7B_CodeTool_1node.sh" \
    ++actor_rollout_ref.ref.model_path="${MATH_TEACHER_MODEL_PATH}" \
    ++actor_rollout_ref.ref.math_model_path="${MATH_TEACHER_MODEL_PATH}" \
    ++actor_rollout_ref.ref.code_model_path="${CODE_TEACHER_MODEL_PATH}" \
    ++actor_rollout_ref.actor.use_routed_opd_loss=True \
    ++actor_rollout_ref.actor.routed_opd_loss_mode="${ROUTED_OPD_LOSS_MODE}" \
    ++actor_rollout_ref.actor.routed_opd_loss_coef="${ROUTED_OPD_LOSS_COEF}" \
    ++actor_rollout_ref.actor.routed_opd_use_task_rewards="${ROUTED_OPD_USE_TASK_REWARDS}" \
    ++actor_rollout_ref.actor.routed_opd_use_policy_gradient="${ROUTED_OPD_USE_POLICY_GRADIENT}" \
    ++actor_rollout_ref.actor.routed_opd_loss_max_clamp="${ROUTED_OPD_LOSS_MAX_CLAMP}" \
    "$@"
