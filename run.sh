#!/usr/bin/env bash
# X-Stream inference entrypoint. Single + multi-checkpoint, vLLM + API models,
# resume support, all multi-stream / token-reduction modes from upstream.
#
# Quickstart:
#   bash run.sh --model Qwen3-Omni-30B-A3B-Instruct \
#               --input tests/sample_10_merged.jsonl \
#               --multi-stream pixel
#
# See README.md for a full flag reference.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pipeline.sh
source "${SCRIPT_DIR}/pipeline.sh"

# ----------------------------------------------------------------------------
usage() {
  cat <<'EOF'
Usage: bash run.sh [options]

Required:
  --model NAME              Logical model name in configs/models.json
                            (must match a top-level key; optional with --warm-cache-only)
  --input JSONL             Input task list (one JSON object per line)

Run identity / output:
  --run-id ID               Run identifier (default: demo)
  --output-dir DIR          Output root for RUN_DIRs (default: ./outputs)
  --config FILE             models.json path (default: configs/models.json)

Concurrency / vLLM:
  --workers N               mllmflow worker count (default: 4)
  --tp N                    vLLM tensor parallel size per instance (default: 2)
  --vllm-model-path DIR     Local checkpoint root used by vLLM
                            (default: $VLLM_MODEL_PATH or ./checkpoints)
  --no-vllm                 API-only models: do not start a local vLLM service
  --gpu-mem-util F          vLLM --gpu-memory-utilization (default: 0.85)
  --max-model-len N         vLLM --max-model-len (optional)

Multi-checkpoint (loops one full run per checkpoint):
  --ckpt PATH               Repeatable; each PATH gets its own run
  --ckpt-list-file FILE     One path per line ('#' and blank lines ignored)

Resume:
  --resume                  Reuse the newest matching incomplete RUN_DIR

Multi-stream / token reduction:
  --multi-stream MODE       pixel | time | code | code_adaptive
                            | cdpruner | surge
                            | cdpruner_token | surge_token
                            (default: pixel)
  --surge-rho FLOAT         FLOW_SURGE_RHO for --multi-stream surge (default: 0.75)
  --cdpruner-keep-ratio F   FLOW_CDPRUNER_KEEP_RATIO for --multi-stream cdpruner
                            (default: 0.5)
  --xstream-rho FLOAT       Patch-level pruning rate for cdpruner_token / surge_token
                            (default: 0.25, range [0,1); local vLLM only)

Inputs / IO:
  --prompt-root DIR         Root for {{file:...}} system prompts
  --video-root DIR          Root for {{video:...}} resources
  --image-root DIR          Root for {{image:...}} resources
  --cache-dir DIR           Decoded-frame / segment cache (default: ./cache)
  --warm-cache-only         Pre-generate video segment cache and exit (no vLLM/model calls)
  --cache-warm-workers N    Worker count for --warm-cache-only (default: --workers)
  --drop-audio              Drop all audio inputs inside mllmflow
  --use-audio-in-video      Ask vLLM/Qwen-Omni to extract audio from video_url inputs
  --api-timeout SECS        Per-request timeout (default: 600)

Eval:
  --stream-eval             Run stream-eval after inference (default: on)
  --no-stream-eval          Skip stream-eval; only produce raw output JSONL
  --stream-eval-judger M    Judger model name (default: qwen3-235b-a22b-instruct-2507)

Other:
  -h, --help                Show this help and exit
EOF
}

# ----------------------------------------------------------------------------
# Defaults (env vars are still honoured if exported by the caller).
ARG_MODEL=""
ARG_INPUT=""
ARG_RUN_ID="${RUN_ID:-demo}"
ARG_OUTPUT_DIR="${FLOW_OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"
ARG_CONFIG="${FLOW_CONFIG:-${SCRIPT_DIR}/configs/models.json}"
ARG_CONFIG_EXPLICIT=0
ARG_WORKERS="${FLOW_N_WORKERS:-4}"
ARG_TP="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
ARG_VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-${SCRIPT_DIR}/checkpoints}"
ARG_NO_VLLM=0
ARG_GPU_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
ARG_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
ARG_RESUME=0
ARG_MULTI_STREAM="${FLOW_MULTI_STREAM_MODE:-pixel}"
ARG_SURGE_RHO="${FLOW_SURGE_RHO:-0.75}"
ARG_CDPRUNER_RATIO="${FLOW_CDPRUNER_KEEP_RATIO:-0.5}"
ARG_XSTREAM_RHO="${XSTREAM_VLLM_PRUNER_RHO:-0.25}"
ARG_PROMPT_ROOT="${FLOW_PROMPT_ROOT:-}"
ARG_VIDEO_ROOT="${FLOW_VIDEO_ROOT:-}"
ARG_IMAGE_ROOT="${FLOW_IMAGE_ROOT:-}"
ARG_CACHE_DIR="${FLOW_CACHE_DIR:-${SCRIPT_DIR}/cache}"
ARG_WARM_CACHE_ONLY=0
ARG_CACHE_WARM_WORKERS="${FLOW_CACHE_WARM_WORKERS:-}"
ARG_DROP_AUDIO="${FLOW_DROP_AUDIO:-False}"
ARG_USE_AUDIO_IN_VIDEO="${FLOW_USE_AUDIO_IN_VIDEO:-False}"
ARG_API_TIMEOUT="${FLOW_API_TIMEOUT:-600}"
ARG_STREAM_EVAL="${ENABLE_STREAM_EVAL:-true}"
ARG_STREAM_EVAL_JUDGER="${STREAM_EVAL_JUDGER:-qwen3-235b-a22b-instruct-2507}"

CKPT_ARGS=()
CKPT_LIST_FILE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --model)              ARG_MODEL="$2"; shift 2 ;;
    --input)              ARG_INPUT="$2"; shift 2 ;;
    --run-id)             ARG_RUN_ID="$2"; shift 2 ;;
    --output-dir)         ARG_OUTPUT_DIR="$2"; shift 2 ;;
    --config)             ARG_CONFIG="$2"; ARG_CONFIG_EXPLICIT=1; shift 2 ;;
    --workers)            ARG_WORKERS="$2"; shift 2 ;;
    --tp)                 ARG_TP="$2"; shift 2 ;;
    --vllm-model-path)    ARG_VLLM_MODEL_PATH="$2"; shift 2 ;;
    --no-vllm)            ARG_NO_VLLM=1; shift ;;
    --gpu-mem-util)       ARG_GPU_MEM_UTIL="$2"; shift 2 ;;
    --max-model-len)      ARG_MAX_MODEL_LEN="$2"; shift 2 ;;
    --ckpt)               CKPT_ARGS+=("$2"); shift 2 ;;
    --ckpt-list-file)     CKPT_LIST_FILE="$2"; shift 2 ;;
    --resume)             ARG_RESUME=1; shift ;;
    --multi-stream)       ARG_MULTI_STREAM="$2"; shift 2 ;;
    --surge-rho)          ARG_SURGE_RHO="$2"; shift 2 ;;
    --cdpruner-keep-ratio) ARG_CDPRUNER_RATIO="$2"; shift 2 ;;
    --xstream-rho)        ARG_XSTREAM_RHO="$2"; shift 2 ;;
    --prompt-root)        ARG_PROMPT_ROOT="$2"; shift 2 ;;
    --video-root)         ARG_VIDEO_ROOT="$2"; shift 2 ;;
    --image-root)         ARG_IMAGE_ROOT="$2"; shift 2 ;;
    --cache-dir)          ARG_CACHE_DIR="$2"; shift 2 ;;
    --warm-cache-only)    ARG_WARM_CACHE_ONLY=1; shift ;;
    --cache-warm-workers) ARG_CACHE_WARM_WORKERS="$2"; shift 2 ;;
    --drop-audio)         ARG_DROP_AUDIO="True"; shift ;;
    --use-audio-in-video) ARG_USE_AUDIO_IN_VIDEO="True"; shift ;;
    --api-timeout)        ARG_API_TIMEOUT="$2"; shift 2 ;;
    --stream-eval)        ARG_STREAM_EVAL="true"; shift ;;
    --no-stream-eval)     ARG_STREAM_EVAL="false"; shift ;;
    --stream-eval-judger) ARG_STREAM_EVAL_JUDGER="$2"; shift 2 ;;
    -h|--help)            usage; exit 0 ;;
    *)                    echo "run.sh: unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

# ----------------------------------------------------------------------------
# Validation
[ -z "${ARG_INPUT}" ] && { echo "Error: --input is required" >&2; exit 1; }
[ -f "${ARG_INPUT}" ] || { echo "Error: --input not found: ${ARG_INPUT}" >&2; exit 1; }
[ -f "${ARG_CONFIG}" ] || { echo "Error: --config not found: ${ARG_CONFIG}" >&2; exit 1; }
if [ -z "${ARG_MODEL}" ]; then
  if [ "${ARG_WARM_CACHE_ONLY}" -eq 1 ]; then
    ARG_MODEL="__warm_cache_only__"
  else
    echo "Error: --model is required" >&2
    exit 1
  fi
fi

# pipeline.sh cd's into RUN_DIR before launching mllmflow, so all path-like
# arguments must be absolute.
abspath() { python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"; }
ARG_INPUT="$(abspath "${ARG_INPUT}")"
ARG_CONFIG="$(abspath "${ARG_CONFIG}")"
ARG_OUTPUT_DIR="$(abspath "${ARG_OUTPUT_DIR}")"
ARG_CACHE_DIR="$(abspath "${ARG_CACHE_DIR}")"
[ -n "${ARG_PROMPT_ROOT}" ] && ARG_PROMPT_ROOT="$(abspath "${ARG_PROMPT_ROOT}")"
[ -n "${ARG_VIDEO_ROOT}"  ] && ARG_VIDEO_ROOT="$(abspath "${ARG_VIDEO_ROOT}")"
[ -n "${ARG_IMAGE_ROOT}"  ] && ARG_IMAGE_ROOT="$(abspath "${ARG_IMAGE_ROOT}")"
[ -n "${ARG_VLLM_MODEL_PATH}" ] && ARG_VLLM_MODEL_PATH="$(abspath "${ARG_VLLM_MODEL_PATH}")"

case "${ARG_MULTI_STREAM}" in
  pixel|time|code|code_adaptive|cdpruner|surge) ;;
  cdpruner_token|surge_token)
    # Token-level patch pruning requires a local vLLM worker so the
    # xstream_vllm_pruner plugin can install itself.
    if [ "${ARG_NO_VLLM}" -eq 1 ] && [ "${ARG_WARM_CACHE_ONLY}" -ne 1 ]; then
      echo "Error: --multi-stream ${ARG_MULTI_STREAM} requires a local vLLM backend; remove --no-vllm." >&2
      exit 1
    fi
    ;;
  *) echo "Error: --multi-stream must be one of pixel,time,code,code_adaptive,cdpruner,surge,cdpruner_token,surge_token" >&2; exit 1 ;;
esac

# ----------------------------------------------------------------------------
# Build the checkpoint loop list. Three sources, in priority order:
#   1) --ckpt (repeatable)
#   2) --ckpt-list-file
#   3) $VLLM_SERVE_MODEL (single ckpt) or empty (use --vllm-model-path/--model)
CKPTS=()
if [ "${#CKPT_ARGS[@]}" -gt 0 ]; then
  CKPTS=("${CKPT_ARGS[@]}")
elif [ -n "${CKPT_LIST_FILE}" ]; then
  [ -f "${CKPT_LIST_FILE}" ] || { echo "Error: --ckpt-list-file not found: ${CKPT_LIST_FILE}" >&2; exit 1; }
  while IFS= read -r line || [ -n "${line}" ]; do
    line="${line#"${line%%[![:space:]]*}"}"; line="${line%"${line##*[![:space:]]}"}"
    [ -z "${line}" ] && continue
    [ "${line:0:1}" = "#" ] && continue
    CKPTS+=("${line}")
  done < "${CKPT_LIST_FILE}"
elif [ -n "${VLLM_SERVE_MODEL:-}" ]; then
  CKPTS=("${VLLM_SERVE_MODEL}")
else
  CKPTS=("")  # empty -> rely on VLLM_MODEL_PATH/VLLM_MODEL_NAME resolution
fi
[ "${#CKPTS[@]}" -eq 0 ] && { echo "Error: empty checkpoint list" >&2; exit 1; }
N_CKPT="${#CKPTS[@]}"

# When user supplied an explicit ckpt list, default to resume-on so finished
# checkpoints are skipped and unfinished ones resume in their existing RUN_DIR.
if { [ "${#CKPT_ARGS[@]}" -gt 0 ] || [ -n "${CKPT_LIST_FILE}" ]; } && [ "${ARG_RESUME}" -eq 0 ]; then
  ARG_RESUME=1
  echo "Multi-checkpoint mode: --resume defaulted to on (finished ckpts will be skipped)."
fi

# ----------------------------------------------------------------------------
# Export environment for pipeline.sh / mllmflow.
export RUN_ID="${ARG_RUN_ID}"
export FLOW_OUTPUT_DIR="${ARG_OUTPUT_DIR}"
export FLOW_CONFIG="${ARG_CONFIG}"
export FLOW_INPUT="${ARG_INPUT}"
export FLOW_N_WORKERS="${ARG_WORKERS}"
export VLLM_TENSOR_PARALLEL_SIZE="${ARG_TP}"
export VLLM_MODEL_PATH="${ARG_VLLM_MODEL_PATH}"
export VLLM_MODEL_NAME="${ARG_MODEL}"
export VLLM_GPU_MEMORY_UTILIZATION="${ARG_GPU_MEM_UTIL}"
[ -n "${ARG_MAX_MODEL_LEN}" ] && export VLLM_MAX_MODEL_LEN="${ARG_MAX_MODEL_LEN}"
export FLOW_MULTI_STREAM_MODE="${ARG_MULTI_STREAM}"
export FLOW_SURGE_RHO="${ARG_SURGE_RHO}"
export FLOW_CDPRUNER_KEEP_RATIO="${ARG_CDPRUNER_RATIO}"
export XSTREAM_VLLM_PRUNER_RHO="${ARG_XSTREAM_RHO}"
export FLOW_PROMPT_ROOT="${ARG_PROMPT_ROOT}"
export FLOW_VIDEO_ROOT="${ARG_VIDEO_ROOT}"
export FLOW_IMAGE_ROOT="${ARG_IMAGE_ROOT}"
export FLOW_CACHE_DIR="${ARG_CACHE_DIR}"
if [ "${ARG_WARM_CACHE_ONLY}" -eq 1 ]; then
  export FLOW_WARM_CACHE_ONLY="true"
  export ENABLE_STREAM_EVAL="false"
  export ENABLE_VLLM_SERVICES="false"
else
  export FLOW_WARM_CACHE_ONLY="${FLOW_WARM_CACHE_ONLY:-false}"
fi
[ -n "${ARG_CACHE_WARM_WORKERS}" ] && export FLOW_CACHE_WARM_WORKERS="${ARG_CACHE_WARM_WORKERS}"
export FLOW_DROP_AUDIO="${ARG_DROP_AUDIO}"
export FLOW_USE_AUDIO_IN_VIDEO="${ARG_USE_AUDIO_IN_VIDEO}"
export FLOW_API_TIMEOUT="${ARG_API_TIMEOUT}"
export FLOW_REPLACEMENT="MODEL>${ARG_MODEL}"
if [ "${ARG_WARM_CACHE_ONLY}" -ne 1 ]; then
  export ENABLE_STREAM_EVAL="${ARG_STREAM_EVAL}"
fi
export STREAM_EVAL_JUDGER="${ARG_STREAM_EVAL_JUDGER}"
if [ "${ARG_WARM_CACHE_ONLY}" -eq 1 ]; then
  export ENABLE_VLLM_SERVICES="false"
elif [ "${ARG_NO_VLLM}" -eq 1 ]; then
  export ENABLE_VLLM_SERVICES="false"
else
  export ENABLE_VLLM_SERVICES="${ENABLE_VLLM_SERVICES:-true}"
fi

mkdir -p "${ARG_OUTPUT_DIR}" "${ARG_CACHE_DIR}"

# ----------------------------------------------------------------------------
# SIGINT/SIGTERM: stop vLLM PIDs we own and exit promptly.
trap 'cleanup_vllm; echo "Interrupted."; exit 130' SIGINT SIGTERM

# Best-effort kill of stale vLLM processes from a previous round.
cleanup_vllm_between_rounds() {
  [ "${ENABLE_VLLM_SERVICES}" != "true" ] && return 0
  if [ -n "${PREV_RUN_DIR:-}" ] && [ -f "${PREV_RUN_DIR}/vllm_pids.txt" ]; then
    local pids
    pids=$(tr ' ' '\n' < "${PREV_RUN_DIR}/vllm_pids.txt" | grep -E '^[0-9]+$' | sort -u || true)
    [ -n "${pids}" ] && echo "${pids}" | xargs -r kill -9 2>/dev/null || true
  fi
  command -v pkill >/dev/null 2>&1 && pkill -9 -f "tools/vllm_cli.py" 2>/dev/null || true
  sleep 2
}

# ----------------------------------------------------------------------------
# Run loop
RUN_ID_BASE="${RUN_ID}"
PREV_RUN_DIR=""
FLOW_CONFIG_INITIAL="${FLOW_CONFIG}"
VLLM_PORT_LIST_INITIAL="${VLLM_PORT_LIST:-}"
VLLM_CUDA_DEVICES_LIST_INITIAL="${VLLM_CUDA_DEVICES_LIST:-}"

run_one_ckpt() {
  unset CONTINUE_RESUMING RUN_DIR RUN_ID_WITH_TIMESTAMP FLOW_OUTPUT
  unset VLLM_PORT_LIST VLLM_CUDA_DEVICES_LIST
  [ -n "${VLLM_PORT_LIST_INITIAL}" ] && export VLLM_PORT_LIST="${VLLM_PORT_LIST_INITIAL}"
  [ -n "${VLLM_CUDA_DEVICES_LIST_INITIAL}" ] && export VLLM_CUDA_DEVICES_LIST="${VLLM_CUDA_DEVICES_LIST_INITIAL}"
  export FLOW_CONFIG="${FLOW_CONFIG_INITIAL}"

  if [ "${ARG_RESUME}" -eq 1 ]; then
    local status_line status status_dir
    status_line=$(python3 "${SCRIPT_DIR}/tools/continue_run_status.py" status 2>/dev/null || echo "none")
    status="${status_line%%$'\t'*}"
    status_dir="${status_line#*$'\t'}"
    case "${status}" in
      complete)
        echo "Resume: matching run is already complete, skipping: ${status_dir}"
        return 0
        ;;
      resume)
        eval "$(python3 "${SCRIPT_DIR}/tools/print_continue_exports.py" "${status_dir}")"
        if [ "${ARG_NO_VLLM}" -eq 1 ] || [ "${ARG_CONFIG_EXPLICIT}" -eq 1 ]; then
          export FLOW_CONFIG="${FLOW_CONFIG_INITIAL}"
          echo "Resume: using current FLOW_CONFIG=${FLOW_CONFIG}"
        fi
        export CONTINUE_RESUMING=1
        echo "Resume: reusing RUN_DIR=${RUN_DIR}"
        ;;
      *)
        echo "Resume: no matching run found, starting fresh."
        ;;
    esac
  fi

  if ! phase1_start_vllm; then
    echo "Error: phase1_start_vllm failed. See ${RUN_DIR:-RUN_DIR}/vllmlogs/" >&2
    return 1
  fi

  if [ -z "${RUN_DIR:-}" ] || [ ! -d "${RUN_DIR}" ]; then
    echo "Error: RUN_DIR not set after phase1." >&2
    return 1
  fi

  case "${FLOW_CONFIG}" in
    /*) ;;
    *) export FLOW_CONFIG="${SCRIPT_DIR}/${FLOW_CONFIG}" ;;
  esac
  ( cd "${RUN_DIR}" && phase2_run_flow )
}

i=0
for ckpt in "${CKPTS[@]}"; do
  if [ "${ENABLE_VLLM_SERVICES}" = "true" ] && [ -n "${ckpt}" ] && [ ! -d "${ckpt}" ]; then
    echo "Error: --ckpt path is not a directory: ${ckpt}" >&2
    exit 1
  fi

  if [ "${i}" -gt 0 ]; then
    echo "[round $((i+1))/${N_CKPT}] cleaning up previous vLLM processes ..."
    cleanup_vllm_between_rounds
  fi

  if [ -n "${ckpt}" ]; then
    export VLLM_SERVE_MODEL="${ckpt}"
    bn="$(basename "${ckpt}")"
    bn="${bn//[^a-zA-Z0-9._-]/_}"
    if [ "${N_CKPT}" -gt 1 ]; then
      export RUN_ID="${RUN_ID_BASE}__${bn}"
    else
      export RUN_ID="${RUN_ID_BASE}"
    fi
  else
    unset VLLM_SERVE_MODEL
    export RUN_ID="${RUN_ID_BASE}"
  fi

  echo "========== checkpoint $((i+1))/${N_CKPT}: ${ckpt:-<auto>} (RUN_ID=${RUN_ID}) =========="
  if ! run_one_ckpt; then
    echo "Round failed; aborting subsequent checkpoints." >&2
    exit 1
  fi
  PREV_RUN_DIR="${RUN_DIR:-}"
  i=$((i + 1))
done

echo "Done."
