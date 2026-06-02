#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
echo "Switched to directory: ${SCRIPT_DIR}"

mkdir -p logs
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"

CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/config/qwen2.5_7b_aime24_code_only.env}"
if [[ -f "${CONFIG_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${CONFIG_FILE}"
else
    echo "Config file not found: ${CONFIG_FILE}" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CODE_TOOL_PYTHON_BIN="${CODE_TOOL_PYTHON_BIN:-${PYTHON_BIN}}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/qwen2.5_7b_aime24_code_only}"
DATASET_NAME="${DATASET_NAME:-aime24}"
TURNS="${TURNS:-1}"
CHECK_ENDPOINT_READY="${CHECK_ENDPOINT_READY:-true}"

if [[ "${CHECK_ENDPOINT_READY}" == "true" ]]; then
    FIRST_API_KEY="${API_KEYS:-EMPTY}"
    FIRST_API_KEY="${FIRST_API_KEY%% *}"
    for endpoint in ${ENDPOINTS}; do
        models_url="${endpoint%/}/models"
        echo "Checking vLLM endpoint: ${models_url}"
        if ! curl --noproxy '*' -fsS -H "Authorization: Bearer ${FIRST_API_KEY}" "${models_url}" >/dev/null; then
            echo "vLLM endpoint is not reachable: ${models_url}" >&2
            echo "Start it first with: bash ${SCRIPT_DIR}/deploy_qwen2.5_7B.sh" >&2
            echo "Then verify with: curl --noproxy '*' -H 'Authorization: Bearer ${FIRST_API_KEY}' ${models_url}" >&2
            exit 1
        fi
    done
fi

INFER_CMD=(
    "${PYTHON_BIN}" -u infer_code_only.py
    --endpoints ${ENDPOINTS}
    --model_path "${MODEL_PATH}"
    --default_model "${DEFAULT_MODEL}"
    --dataset_name "${DATASET_NAME}"
    --data_path "${DATA_PATH}"
    --output_path "${OUTPUT_PATH}"
    --turns ${TURNS}
    --counts "${COUNTS}"
    --max_concurrent_requests "${MAX_CONCURRENT_REQUESTS}"
    --sample_timeout "${SAMPLE_TIMEOUT}"
    --temperature "${TEMPERATURE}"
    --max_tokens "${MAX_TOKENS}"
    --top_p "${TOP_P}"
    --repetition_penalty "${REPETITION_PENALTY}"
    --python_bin "${CODE_TOOL_PYTHON_BIN}"
    --python_max_concurrent "${PYTHON_MAX_CONCURRENT}"
    --python_timeout "${PYTHON_TIMEOUT}"
    --max_python_times "${MAX_PYTHON_TIMES}"
)

if [[ -n "${API_KEYS:-}" ]]; then
    INFER_CMD+=(--api_keys ${API_KEYS})
fi

echo "Running inference:"
printf ' %q' "${INFER_CMD[@]}"
echo
"${INFER_CMD[@]}" 2>&1 | tee "logs/qwen2.5_7b_aime24_code_only_infer.log"

if [[ "${RUN_EVAL:-true}" == "true" ]]; then
    for turn in ${TURNS}; do
        OUTPUT_FILE="${OUTPUT_PATH}/${DATASET_NAME}/${DATASET_NAME}_output_${turn}.json"
        EVAL_CMD=(
            "${PYTHON_BIN}" -u evaluate.py
            --output_path "${OUTPUT_FILE}"
            --task math
            --concurrent_limit "${EVAL_CONCURRENT_LIMIT}"
            --timeout "${EVAL_TIMEOUT}"
        )
        if [[ "${EVAL_USE_LLM:-false}" == "true" ]]; then
            EVAL_CMD+=(--use_llm)
            EVAL_CMD+=(--api_base_url "${EVAL_API_BASE_URL}")
            EVAL_CMD+=(--model_name "${EVAL_MODEL_NAME}")
        fi

        echo "Running evaluation for turn ${turn}:"
        printf ' %q' "${EVAL_CMD[@]}"
        echo
        "${EVAL_CMD[@]}" 2>&1 | tee "logs/qwen2.5_7b_aime24_code_only_eval_turn_${turn}.log"
    done
fi
