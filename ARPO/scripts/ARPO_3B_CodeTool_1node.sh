#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ARPO RL with code tool for Qwen2.5-3B-Instruct.
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-/workspace/hf/Qwen/Qwen2.5-3B-Instruct}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-arpo_3b_code_tool_only}"

exec bash "${SCRIPT_DIR}/ARPO_7B_CodeTool_1node.sh" "$@"
