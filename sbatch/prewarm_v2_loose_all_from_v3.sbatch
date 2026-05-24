#!/bin/bash
#SBATCH -J STRM-v2-loose-prewarm
#SBATCH -N 1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=12G
#SBATCH -o /home/dyvm6xra/dyvm6xrauser04/peiwensun/project/X-Stream-open-source/inference/logs/%x_%j.out
#SBATCH -e /home/dyvm6xra/dyvm6xrauser04/peiwensun/project/X-Stream-open-source/inference/logs/%x_%j.err

set -euo pipefail

echo "JobID: ${SLURM_JOB_ID}"
echo "NodeList: ${SLURM_NODELIST}"
echo "CONDA_EXE: ${CONDA_EXE:-}"

# sbatch -A dyvm6xra -p q-dyvm6xra -q qos_g5_24hrs -t 24:00:00 sbatch/prewarm_v2_loose_all_from_v3.sbatch

INFERENCE_DIR="/home/dyvm6xra/dyvm6xrauser04/peiwensun/project/X-Stream-open-source/inference"
INPUT_ROOT="/home/dyvm6xra/dyvm6xrauser04/peiwensun/project/X-Stream-open-source/data/v2/loose"
VIDEO_ROOT="/home/dyvm6xra/dyvm6xrauser04/peiwensun/project/X-Stream-open-source/data/v3"
CACHE_DIR="${INFERENCE_DIR}/cache"
WORKERS="${XSTREAM_PREWARM_WORKERS:-80}"

cd "${INFERENCE_DIR}"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

uv run bash run.sh \
  --input "${INPUT_ROOT}/eval_relative_merged_phostream_type.jsonl" \
  --warm-cache-only \
  --workers "${WORKERS}" \
  --cache-warm-workers "${WORKERS}" \
  --cache-dir "${CACHE_DIR}" \
  --run-id prewarm_v2_loose_merged_from_v3 \
  --video-root "${VIDEO_ROOT}"

uv run bash run.sh \
  --input "${INPUT_ROOT}/eval_relative_multi_phostream_type.jsonl" \
  --warm-cache-only \
  --workers "${WORKERS}" \
  --cache-warm-workers "${WORKERS}" \
  --cache-dir "${CACHE_DIR}" \
  --run-id prewarm_v2_loose_multi_from_v3 \
  --video-root "${VIDEO_ROOT}"

echo "Done."
