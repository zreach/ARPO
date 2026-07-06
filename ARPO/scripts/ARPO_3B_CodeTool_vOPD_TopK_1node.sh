#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# vOPD top-k from "KL for a KL": base OPD reward plus a detached top-k
# reverse-KL control variate baseline. Default k=20 follows the paper.
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-/workspace/hf/Qwen/Qwen2.5-3B-Instruct}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/workspace/hf/Qwen/Qwen2.5-7B-Instruct}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-vopd_topk_3b_code_tool_only}"

export USE_OPD_LOSS="${USE_OPD_LOSS:-False}"
export OPD_LOSS_COEF="${OPD_LOSS_COEF:-0.0}"

VOPD_TOPK="${VOPD_TOPK:-20}"
VOPD_LOSS_COEF="${VOPD_LOSS_COEF:-1.0}"
VOPD_USE_TASK_REWARDS="${VOPD_USE_TASK_REWARDS:-False}"
VOPD_USE_POLICY_GRADIENT="${VOPD_USE_POLICY_GRADIENT:-True}"
VOPD_REWARD_MAX_CLAMP="${VOPD_REWARD_MAX_CLAMP:-10.0}"

exec bash "${SCRIPT_DIR}/ARPO_3B_CodeTool_1node.sh" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    ++actor_rollout_ref.ref.model_path="${TEACHER_MODEL_PATH}" \
    ++actor_rollout_ref.actor.export_vopd_topk=True \
    ++actor_rollout_ref.actor.vopd_topk="${VOPD_TOPK}" \
    ++actor_rollout_ref.actor.use_vopd_loss=True \
    ++actor_rollout_ref.actor.vopd_loss_coef="${VOPD_LOSS_COEF}" \
    ++actor_rollout_ref.actor.vopd_use_task_rewards="${VOPD_USE_TASK_REWARDS}" \
    ++actor_rollout_ref.actor.vopd_use_policy_gradient="${VOPD_USE_POLICY_GRADIENT}" \
    ++actor_rollout_ref.actor.vopd_reward_max_clamp="${VOPD_REWARD_MAX_CLAMP}" \
    "$@"
