#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Baseline RL: keep GRPO + code tool, but disable ARPO adaptive branching.
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-/workspace/hf/Qwen/Qwen3-1.7B}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-baseline_1_7b_code_tool_grpo}"
export ROLLOUT_N="${ROLLOUT_N:-16}"
export INITIAL_ROLLOUTS="${INITIAL_ROLLOUTS:-${ROLLOUT_N}}"
export BEAM_SIZE="${BEAM_SIZE:-1}"
export BRANCH_PROBABILITY="${BRANCH_PROBABILITY:-0}"
export ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-0}"

exec bash "${SCRIPT_DIR}/ARPO_7B_CodeTool_1node.sh" "$@"
