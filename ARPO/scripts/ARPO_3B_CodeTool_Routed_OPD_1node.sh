#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-/workspace/hf/Qwen/Qwen2.5-3B-Instruct}"
MATH_TEACHER_MODEL_PATH="${MATH_TEACHER_MODEL_PATH:-/workspace/hf/Qwen/Qwen2.5-Math-7B-Instruct}"
CODE_TEACHER_MODEL_PATH="${CODE_TEACHER_MODEL_PATH:-/workspace/hf/Qwen/Qwen2.5-Coder-7B-Instruct}"

export ACTOR_MODEL_PATH
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-routed_opd_3b_code_tool_only}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.92}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-1000}"
export VERL_CPU_MEMORY_GUARD_MAX_USED_RATIO="${VERL_CPU_MEMORY_GUARD_MAX_USED_RATIO:-0.92}"
export VERL_CPU_MEMORY_GUARD_MIN_AVAILABLE_GB="${VERL_CPU_MEMORY_GUARD_MIN_AVAILABLE_GB:-32}"
export VERL_CPU_GC_INTERVAL_STEPS="${VERL_CPU_GC_INTERVAL_STEPS:-1}"
export VERL_CPU_GC_TRIGGER_USED_RATIO="${VERL_CPU_GC_TRIGGER_USED_RATIO:-0.85}"
export VERL_USE_FLASH_ATTN_CROSS_ENTROPY="${VERL_USE_FLASH_ATTN_CROSS_ENTROPY:-FALSE}"
export VERL_GPU_MEMORY_GUARD_MAX_USED_RATIO="${VERL_GPU_MEMORY_GUARD_MAX_USED_RATIO:-0.94}"
export VERL_GPU_MEMORY_GUARD_MIN_FREE_GB="${VERL_GPU_MEMORY_GUARD_MIN_FREE_GB:-4}"

# Keep the earlier lightweight OPD branch disabled; this script uses routed OPD.
export USE_OPD_LOSS="${USE_OPD_LOSS:-False}"
export OPD_LOSS_COEF="${OPD_LOSS_COEF:-0.0}"

ROUTED_OPD_LOSS_MODE="${ROUTED_OPD_LOSS_MODE:-k1}"
ROUTED_OPD_LOSS_COEF="${ROUTED_OPD_LOSS_COEF:-1.0}"
ROUTED_OPD_USE_TASK_REWARDS="${ROUTED_OPD_USE_TASK_REWARDS:-True}"
ROUTED_OPD_USE_POLICY_GRADIENT="${ROUTED_OPD_USE_POLICY_GRADIENT:-True}"
ROUTED_OPD_LOSS_MAX_CLAMP="${ROUTED_OPD_LOSS_MAX_CLAMP:-10.0}"

exec "${SCRIPT_DIR}/ARPO_3B_CodeTool_1node.sh" \
    actor_rollout_ref.actor.use_kl_loss=False \
    ++actor_rollout_ref.ref.model_path="${MATH_TEACHER_MODEL_PATH}" \
    ++actor_rollout_ref.ref.math_model_path="${MATH_TEACHER_MODEL_PATH}" \
    ++actor_rollout_ref.ref.code_model_path="${CODE_TEACHER_MODEL_PATH}" \
    ++actor_rollout_ref.ref.skip_default_ref_log_prob=True \
    ++actor_rollout_ref.actor.use_routed_opd_loss=True \
    ++actor_rollout_ref.actor.routed_opd_loss_mode="${ROUTED_OPD_LOSS_MODE}" \
    ++actor_rollout_ref.actor.routed_opd_loss_coef="${ROUTED_OPD_LOSS_COEF}" \
    ++actor_rollout_ref.actor.routed_opd_use_task_rewards="${ROUTED_OPD_USE_TASK_REWARDS}" \
    ++actor_rollout_ref.actor.routed_opd_use_policy_gradient="${ROUTED_OPD_USE_POLICY_GRADIENT}" \
    ++actor_rollout_ref.actor.routed_opd_loss_max_clamp="${ROUTED_OPD_LOSS_MAX_CLAMP}" \
    "$@"
