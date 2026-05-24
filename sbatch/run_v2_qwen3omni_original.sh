#!/usr/bin/env bash

set -euo pipefail

: "${XSTREAM_MODE:?Set XSTREAM_MODE to a run.sh --multi-stream mode}"

INFERENCE_DIR="/home/dyvm6xra/dyvm6xrauser04/peiwensun/project/X-Stream-open-source/inference"
PROJECT_DIR="/home/dyvm6xra/dyvm6xrauser04/peiwensun/project"
INPUT_ROOT="${PROJECT_DIR}/X-Stream-open-source/data/v2/loose"
VIDEO_ROOT="${PROJECT_DIR}/X-Stream-open-source/data/v3"
PROMPT_ROOT="${PROJECT_DIR}/VLLMFlow/annotations/system_prompt/streaming_prompt"
VLLM_MODEL_PATH="${PROJECT_DIR}/StreamEval/checkpoint_lib"
FLOW_CONFIG="${XSTREAM_MODEL_CONFIG:-${PROJECT_DIR}/VLLMFlow/projects/configs/models.json}"
STREAM_EVAL_JUDGER="${XSTREAM_STREAM_EVAL_JUDGER:-qwen3-235b-a22b-instruct-2507}"

case "${XSTREAM_MODE}" in
  pixel)
    INPUT_FILE="${INPUT_ROOT}/eval_relative_merged_phostream_type.jsonl"
    ;;
  time|code|code_adaptive|cdpruner|surge|cdpruner_token|surge_token)
    INPUT_FILE="${INPUT_ROOT}/eval_relative_multi_phostream_type.jsonl"
    ;;
  *)
    echo "Unsupported XSTREAM_MODE: ${XSTREAM_MODE}" >&2
    exit 1
    ;;
esac

WORKERS="${XSTREAM_WORKERS:-4}"
EXTRA_ARGS=()
case "${XSTREAM_MODE}" in
  cdpruner_token|surge_token)
    WORKERS="${XSTREAM_WORKERS:-1}"
    EXTRA_ARGS+=(--xstream-rho "${XSTREAM_RHO:-0.25}")
    ;;
esac

RUN_ID="v2_loose_qwen3omni_original_${XSTREAM_MODE}"

echo "JobID: ${SLURM_JOB_ID:-}"
echo "NodeList: ${SLURM_NODELIST:-}"
echo "CONDA_EXE: ${CONDA_EXE:-}"
echo "XSTREAM_MODE: ${XSTREAM_MODE}"
echo "INPUT_FILE: ${INPUT_FILE}"
echo "VIDEO_ROOT: ${VIDEO_ROOT}"
echo "RUN_ID: ${RUN_ID}"
echo "FLOW_CONFIG: ${FLOW_CONFIG}"
echo "STREAM_EVAL_JUDGER: ${STREAM_EVAL_JUDGER}"

cd "${INFERENCE_DIR}"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export FLOW_CONFIG
export STREAM_EVAL_JUDGER

# 可选：打印 GPU 状态
nvidia-smi

uv run bash run.sh \
  --model Qwen3-Omni-30B-A3B-Instruct \
  --vllm-model-path "${VLLM_MODEL_PATH}" \
  --input "${INPUT_FILE}" \
  --multi-stream "${XSTREAM_MODE}" \
  --tp "${XSTREAM_TP:-2}" --workers "${WORKERS}" \
  --max-model-len "${XSTREAM_MAX_MODEL_LEN:-200000}" \
  --run-id "${RUN_ID}" \
  --prompt-root "${PROMPT_ROOT}" \
  --video-root "${VIDEO_ROOT}" \
  "${EXTRA_ARGS[@]}"

echo "Done."
