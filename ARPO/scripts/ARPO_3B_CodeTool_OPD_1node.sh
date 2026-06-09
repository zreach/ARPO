#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-/workspace/hf/Qwen/Qwen2.5-3B-Instruct}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/workspace/ARPO/ARPO/checkpoints/baseline_7b_code_tool_grpo/global_step_78/hf}"

export ACTOR_MODEL_PATH
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd_3b_code_tool_only}"

# Keep the earlier lightweight OPD branch disabled; this script uses single-teacher OPD.
export USE_OPD_LOSS="${USE_OPD_LOSS:-False}"
export OPD_LOSS_COEF="${OPD_LOSS_COEF:-0.0}"

MYVERL_OPD_LOSS_MODE="${MYVERL_OPD_LOSS_MODE:-k1}"
MYVERL_OPD_LOSS_COEF="${MYVERL_OPD_LOSS_COEF:-1.0}"
MYVERL_OPD_USE_TASK_REWARDS="${MYVERL_OPD_USE_TASK_REWARDS:-True}"
MYVERL_OPD_USE_POLICY_GRADIENT="${MYVERL_OPD_USE_POLICY_GRADIENT:-True}"
MYVERL_OPD_LOSS_MAX_CLAMP="${MYVERL_OPD_LOSS_MAX_CLAMP:-10.0}"

exec "${SCRIPT_DIR}/ARPO_3B_CodeTool_1node.sh" \
    ++actor_rollout_ref.ref.model_path="${TEACHER_MODEL_PATH}" \
    ++actor_rollout_ref.actor.use_myverl_opd_loss=True \
    ++actor_rollout_ref.actor.myverl_opd_loss_mode="${MYVERL_OPD_LOSS_MODE}" \
    ++actor_rollout_ref.actor.myverl_opd_loss_coef="${MYVERL_OPD_LOSS_COEF}" \
    ++actor_rollout_ref.actor.myverl_opd_use_task_rewards="${MYVERL_OPD_USE_TASK_REWARDS}" \
    ++actor_rollout_ref.actor.myverl_opd_use_policy_gradient="${MYVERL_OPD_USE_POLICY_GRADIENT}" \
    ++actor_rollout_ref.actor.myverl_opd_loss_max_clamp="${MYVERL_OPD_LOSS_MAX_CLAMP}" \
    "$@"
