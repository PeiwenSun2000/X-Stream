#!/usr/bin/env bash
# pipeline.sh — sourced by run.sh. Provides:
#   phase1_start_vllm        bring up vLLM service(s), wait /health
#   phase2_run_flow          run mllmflow.cli + optional stream-eval, clean up
#   start_vllm_service       launch a single vLLM instance in the background
#   cleanup_vllm             trap handler used on SIGINT/SIGTERM
#   find_resume_run_dir      ask continue_run_status.py for a resumable RUN_DIR
#
# All variables are read from the environment; run.sh exports them.

set -u

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="${PIPELINE_DIR}/tools"

# Best-effort cleanup: kill every vLLM child PID we recorded for this RUN_DIR.
cleanup_vllm() {
  if [ -n "${RUN_DIR:-}" ] && [ -f "${RUN_DIR}/vllm_pids.txt" ]; then
    local pids
    pids=$(tr ' ' '\n' < "${RUN_DIR}/vllm_pids.txt" | grep -E '^[0-9]+$' | sort -u || true)
    if [ -n "${pids}" ]; then
      echo "Stopping vLLM PIDs: ${pids}"
      echo "${pids}" | xargs -r kill -9 2>/dev/null || true
    fi
  fi
}

find_resume_run_dir() {
  python3 "${TOOLS_DIR}/continue_run_status.py" resume-dir
}

get_free_ports() {
  python3 "${TOOLS_DIR}/get_free_ports.py" "$1"
}

# ----------------------------------------------------------------------------
# phase1_start_vllm
#   Resolve RUN_DIR (new or resumed), allocate ports + GPU devices when not
#   provided, generate the per-run models.json, write run_env.json, and start
#   one vLLM instance per (port, device-group). Polls /health until ready.
# ----------------------------------------------------------------------------
phase1_start_vllm() {
  local ts cuda_count service_count i j gpu_start dev_str
  local out port max attempt
  local PORTS=() DEVICES=()

  if [ "${CONTINUE_RESUMING:-}" = "1" ] && [ -n "${RUN_DIR:-}" ]; then
    export RUN_ID_WITH_TIMESTAMP="${RUN_ID_WITH_TIMESTAMP:-$(basename "${RUN_DIR}")}"
    export FLOW_OUTPUT="${FLOW_OUTPUT:-${RUN_DIR}/output_${FLOW_INPUT##*/}}"
    mkdir -p "${RUN_DIR}"
    echo "Resuming run: RUN_ID=${RUN_ID_WITH_TIMESTAMP}, RUN_DIR=${RUN_DIR}"
  else
    ts=$(date +%Y%m%d-%H%M%S)
    export RUN_ID_WITH_TIMESTAMP="${RUN_ID:-run}_${ts}"
    export RUN_DIR="${FLOW_OUTPUT_DIR}/${RUN_ID_WITH_TIMESTAMP}"
    mkdir -p "${RUN_DIR}"
    export FLOW_OUTPUT="${RUN_DIR}/output_${FLOW_INPUT##*/}"
    echo "New run: RUN_ID=${RUN_ID_WITH_TIMESTAMP}, RUN_DIR=${RUN_DIR}"
  fi

  # Auto-allocate ports + devices when caller hasn't pinned them.
  if [ -z "${VLLM_PORT_LIST:-}" ] && [ -z "${VLLM_CUDA_DEVICES_LIST:-}" ] && [ -n "${VLLM_TENSOR_PARALLEL_SIZE:-}" ]; then
    cuda_count=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | grep -c . || true)
    if [ -n "${cuda_count}" ] && [ "${cuda_count}" -gt 0 ]; then
      service_count=$((cuda_count / VLLM_TENSOR_PARALLEL_SIZE))
      if [ "${service_count}" -gt 0 ]; then
        VLLM_PORT_LIST=$(get_free_ports "${service_count}")
        export VLLM_PORT_LIST
        local devs=()
        for i in $(seq 0 $((service_count - 1))); do
          gpu_start=$((i * VLLM_TENSOR_PARALLEL_SIZE))
          dev_str="${gpu_start}"
          for j in $(seq 1 $((VLLM_TENSOR_PARALLEL_SIZE - 1))); do
            dev_str="${dev_str},$((gpu_start + j))"
          done
          devs+=("${dev_str}")
        done
        local _ifs="${IFS}"
        IFS=';'
        VLLM_CUDA_DEVICES_LIST="${devs[*]}"
        IFS="${_ifs}"
        export VLLM_CUDA_DEVICES_LIST
        echo "vLLM auto-alloc: ${service_count} x TP=${VLLM_TENSOR_PARALLEL_SIZE}, ports=${VLLM_PORT_LIST}, devices=${VLLM_CUDA_DEVICES_LIST}"
      fi
    fi
  fi

  # Generate per-run models.json with the local vLLM endpoint(s) wired in.
  if [ "${ENABLE_VLLM_SERVICES:-true}" = "true" ] && [ -n "${VLLM_PORT_LIST:-}" ] && [ -n "${VLLM_MODEL_NAME:-}" ] && [ -n "${FLOW_CONFIG:-}" ]; then
    out="${RUN_DIR}/models.json"
    if ! python3 "${TOOLS_DIR}/process_config.py" \
        "${FLOW_CONFIG}" "${VLLM_MODEL_NAME}" "${VLLM_PORT_LIST}" "${out}"; then
      echo "Error: process_config failed" >&2
      return 1
    fi
    export FLOW_CONFIG="${out}"
  fi

  python3 "${TOOLS_DIR}/write_run_env.py" "${RUN_DIR}/run_env.json" \
    generated_at "$(date +%Y-%m-%dT%H:%M:%S%z)" \
    RUN_ID "${RUN_ID:-}" \
    RUN_ID_WITH_TIMESTAMP "${RUN_ID_WITH_TIMESTAMP}" \
    RUN_DIR "${RUN_DIR}" \
    ENABLE_VLLM_SERVICES "${ENABLE_VLLM_SERVICES:-true}" \
    VLLM_PORT_LIST "${VLLM_PORT_LIST:-}" \
    VLLM_CUDA_DEVICES_LIST "${VLLM_CUDA_DEVICES_LIST:-}" \
    VLLM_TENSOR_PARALLEL_SIZE "${VLLM_TENSOR_PARALLEL_SIZE:-}" \
    VLLM_MODEL_NAME "${VLLM_MODEL_NAME:-}" \
    VLLM_SERVE_MODEL "${VLLM_SERVE_MODEL:-}" \
    VLLM_MODEL_PATH "${VLLM_MODEL_PATH:-}" \
    FLOW_CONFIG "${FLOW_CONFIG:-}" \
    FLOW_OUTPUT_DIR "${FLOW_OUTPUT_DIR:-}" \
    FLOW_INPUT "${FLOW_INPUT:-}" \
    FLOW_OUTPUT "${FLOW_OUTPUT:-}" \
    FLOW_MULTI_STREAM_MODE "${FLOW_MULTI_STREAM_MODE:-pixel}" \
    FLOW_VIDEO_ROOT "${FLOW_VIDEO_ROOT:-}" \
    FLOW_PROMPT_ROOT "${FLOW_PROMPT_ROOT:-}" \
    FLOW_N_WORKERS "${FLOW_N_WORKERS:-}" \
    FLOW_WARM_CACHE_ONLY "${FLOW_WARM_CACHE_ONLY:-false}" \
    FLOW_CACHE_WARM_WORKERS "${FLOW_CACHE_WARM_WORKERS:-}" \
    FLOW_REPLACEMENT "${FLOW_REPLACEMENT:-}" \
    FLOW_CACHE_DIR "${FLOW_CACHE_DIR:-}"

  if [ "${ENABLE_VLLM_SERVICES:-true}" != "true" ]; then
    echo "ENABLE_VLLM_SERVICES=false: skipping vLLM startup (API-only model)."
    return 0
  fi
  if [ -z "${VLLM_PORT_LIST:-}" ] || [ -z "${VLLM_CUDA_DEVICES_LIST:-}" ]; then
    echo "Error: VLLM_PORT_LIST or VLLM_CUDA_DEVICES_LIST is empty after auto-alloc." >&2
    return 1
  fi

  local _ifs="${IFS}"
  IFS=';' read -ra PORTS <<< "${VLLM_PORT_LIST}"
  IFS=';' read -ra DEVICES <<< "${VLLM_CUDA_DEVICES_LIST}"
  IFS="${_ifs}"
  if [ "${#PORTS[@]}" -ne "${#DEVICES[@]}" ]; then
    echo "Error: ports vs devices count mismatch (${#PORTS[@]} vs ${#DEVICES[@]})" >&2
    return 1
  fi

  for i in "${!PORTS[@]}"; do
    if [ -n "${PORTS[$i]}" ] && [ -n "${DEVICES[$i]}" ]; then
      start_vllm_service "${PORTS[$i]}" "${DEVICES[$i]}"
    fi
  done

  max=720
  local interval=10
  echo "Waiting for vLLM /health on ports: ${VLLM_PORT_LIST} (up to $((max * interval))s). Logs: ${RUN_DIR}/vllmlogs/"
  for port in "${PORTS[@]}"; do
    [ -z "${port}" ] && continue
    attempt=0
    while [ "${attempt}" -lt "${max}" ]; do
      if curl -s -f -X GET "http://localhost:${port}/health" >/dev/null 2>&1; then
        echo "vLLM ready on port ${port}."
        break
      fi
      attempt=$((attempt + 1))
      echo "  waiting on port ${port} ... $((attempt * interval))s / $((max * interval))s"
      sleep "${interval}"
    done
    if [ "${attempt}" -ge "${max}" ]; then
      echo "Error: vLLM never became ready on port ${port}. See ${RUN_DIR}/vllmlogs/${port}.log" >&2
      return 1
    fi
  done

  echo "All vLLM services healthy."
  return 0
}

# ----------------------------------------------------------------------------
# start_vllm_service PORT CUDA_DEVICES
#   Launch a single vLLM OpenAI server in the background and append its PID to
#   ${RUN_DIR}/vllm_pids.txt so cleanup_vllm can reap it.
# ----------------------------------------------------------------------------
start_vllm_service() {
  local port="$1" cuda_devices="$2"
  local serve_target work_dir use_served_model_name=0
  local log

  : "${VLLM_MODEL_PATH:?start_vllm_service: VLLM_MODEL_PATH not set}"
  : "${VLLM_MODEL_NAME:?start_vllm_service: VLLM_MODEL_NAME not set}"
  : "${VLLM_TENSOR_PARALLEL_SIZE:?start_vllm_service: VLLM_TENSOR_PARALLEL_SIZE not set}"

  mkdir -p "${RUN_DIR}/vllmlogs"
  log="${RUN_DIR}/vllmlogs/${port}.log"

  # Resolution rules (in order):
  #   1) user-supplied VLLM_SERVE_MODEL points to a checkpoint dir
  #   2) VLLM_MODEL_PATH/VLLM_MODEL_NAME is itself a checkpoint dir
  #   3) VLLM_MODEL_PATH directly contains config.json
  serve_target="${VLLM_SERVE_MODEL:-${VLLM_MODEL_NAME}}"
  work_dir="${VLLM_MODEL_PATH}"
  if [ -n "${VLLM_SERVE_MODEL:-}" ]; then
    work_dir="${VLLM_SERVE_MODEL}"
    use_served_model_name=1
  elif [ -d "${VLLM_MODEL_PATH}/${VLLM_MODEL_NAME}" ]; then
    serve_target="${VLLM_MODEL_NAME}"
  elif [ -f "${VLLM_MODEL_PATH}/config.json" ]; then
    serve_target="${VLLM_MODEL_PATH}"
    use_served_model_name=1
  fi

  {
    echo "=== vLLM port=${port} CUDA_VISIBLE_DEVICES=${cuda_devices} ==="
    echo "cwd=${work_dir} serve=${serve_target} logical=${VLLM_MODEL_NAME} TP=${VLLM_TENSOR_PARALLEL_SIZE}"
  } > "${log}"

  (
    export PORT="${port}"
    export CUDA_VISIBLE_DEVICES="${cuda_devices}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    cd "${work_dir}" || exit 1

    local served_name=()
    [ "${use_served_model_name}" = "1" ] && served_name=(--served-model-name "${VLLM_MODEL_NAME}")

    local chat_kwargs=()
    [ -n "${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:-}" ] && \
      chat_kwargs=(--default-chat-template-kwargs "${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS}")

    local extra=()
    [ -n "${VLLM_MAX_MODEL_LEN:-}" ]          && extra+=(--max-model-len "${VLLM_MAX_MODEL_LEN}")
    [ "${VLLM_ENFORCE_EAGER:-false}" = "true" ] && extra+=(--enforce-eager)
    [ -n "${VLLM_MAX_NUM_SEQS:-}" ]           && extra+=(--max-num-seqs "${VLLM_MAX_NUM_SEQS}")
    [ -n "${VLLM_MAX_NUM_BATCHED_TOKENS:-}" ] && extra+=(--max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}")
    [ -n "${VLLM_KV_CACHE_DTYPE:-}" ]         && extra+=(--kv-cache-dtype "${VLLM_KV_CACHE_DTYPE}")
    [ -n "${VLLM_TOOL_CALL_PARSER:-}" ]       && extra+=(--enable-auto-tool-choice --tool-call-parser "${VLLM_TOOL_CALL_PARSER}")

    # ------------------------------------------------------------------
    # X-Stream patch-level pruner integration.
    # When FLOW_MULTI_STREAM_MODE is cdpruner_token / surge_token, enable
    # the xstream_vllm_pruner plugin inside vLLM workers (env var picked up
    # by tools/vllm_cli.py) and turn on the EVS path via --video-pruning-rate
    # so Qwen2.5-VL / Qwen3-VL actually invoke compute_retention_mask, which
    # the plugin has monkey-patched.
    # ------------------------------------------------------------------
    case "${FLOW_MULTI_STREAM_MODE:-}" in
      cdpruner_token|surge_token)
        export XSTREAM_VLLM_PRUNER=1
        if [ "${FLOW_MULTI_STREAM_MODE}" = "cdpruner_token" ]; then
          export XSTREAM_VLLM_PRUNER_ALGO=cdpruner
        else
          export XSTREAM_VLLM_PRUNER_ALGO=surge
        fi
        export XSTREAM_VLLM_PRUNER_RHO="${XSTREAM_VLLM_PRUNER_RHO:-0.25}"
        export XSTREAM_VLLM_PRUNER_KEEP_FIRST_FRAME="${XSTREAM_VLLM_PRUNER_KEEP_FIRST_FRAME:-1}"
        extra+=(--video-pruning-rate "${XSTREAM_VLLM_PRUNER_RHO}")
        echo "xstream_vllm_pruner: enabled (algo=${XSTREAM_VLLM_PRUNER_ALGO}, rho=${XSTREAM_VLLM_PRUNER_RHO})"
        ;;
    esac

    python3 "${TOOLS_DIR}/vllm_cli.py" serve "${serve_target}" \
      --trust-remote-code --host 0.0.0.0 --port "${PORT}" --dtype bfloat16 \
      --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}" \
      --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.85}" \
      --limit-mm-per-prompt '{"image":10000,"video":10000,"audio":10000}' \
      --allowed-local-media-path / \
      --gdn-prefill-backend triton \
      "${served_name[@]}" "${chat_kwargs[@]}" "${extra[@]}"
  ) >> "${log}" 2>&1 &
  echo $! >> "${RUN_DIR}/vllm_pids.txt"
  sleep 2
}

# ----------------------------------------------------------------------------
# phase2_run_flow
#   Run the mllmflow CLI on FLOW_INPUT and (optionally) stream-eval, then
#   tear down any vLLM PIDs we own. Returns mllmflow's exit code.
# ----------------------------------------------------------------------------
phase2_run_flow() {
  local args=(
    --model-config "${FLOW_CONFIG}"
    --input "${FLOW_INPUT}"
    --output "${FLOW_OUTPUT}"
    --model-replacement "${FLOW_REPLACEMENT}"
    --video-root "${FLOW_VIDEO_ROOT}"
    --prompt-root "${FLOW_PROMPT_ROOT}"
    --n-workers "${FLOW_N_WORKERS}"
    --cache-dir "${FLOW_CACHE_DIR}"
    --multi-stream-mode "${FLOW_MULTI_STREAM_MODE:-pixel}"
  )
  [ -n "${FLOW_IMAGE_ROOT:-}" ] && args+=(--image-root "${FLOW_IMAGE_ROOT}")
  if [ "${FLOW_WARM_CACHE_ONLY:-false}" = "true" ]; then
    args+=(--warm-cache-only)
    [ -n "${FLOW_CACHE_WARM_WORKERS:-}" ] && args+=(--cache-warm-workers "${FLOW_CACHE_WARM_WORKERS}")
  fi
  if [ "${CONTINUE_RESUMING:-}" = "1" ] || [ "${MLLMFLOW_RESUME:-false}" = "true" ]; then
    args+=(--resume)
  fi

  echo "Running mllmflow ..."
  echo "mllmflow ${args[*]}"
  python3 -m mllmflow.cli "${args[@]}"
  local rc=$?

  if [ "${rc}" -ne 0 ]; then
    echo "mllmflow exited with code ${rc}; skipping stream-eval and vLLM teardown so logs stay reachable." >&2
    return "${rc}"
  fi

  if [ "${FLOW_WARM_CACHE_ONLY:-false}" = "true" ]; then
    echo "FLOW_WARM_CACHE_ONLY=true: skipping stream-eval and vLLM teardown."
    return 0
  fi

  if [ "${ENABLE_STREAM_EVAL:-true}" = "true" ]; then
    local judger="${STREAM_EVAL_JUDGER:-qwen3-235b-a22b-instruct-2507}"
    echo "stream-eval --model-config ${FLOW_CONFIG} --judger '${judger}' --model-output ${FLOW_OUTPUT} --output-dir ${RUN_DIR}" \
      > "${RUN_DIR}/eval.sh"
    bash "${RUN_DIR}/eval.sh" || echo "Warning: stream-eval failed (non-fatal); see ${RUN_DIR}/eval.sh" >&2
  fi

  cleanup_vllm
  return 0
}
