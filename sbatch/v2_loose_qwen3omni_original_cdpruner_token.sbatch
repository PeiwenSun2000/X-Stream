#!/bin/bash
#SBATCH -J STRM-v2-loose-cdptok
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-cpu=14G
#SBATCH -o /home/dyvm6xra/dyvm6xrauser04/peiwensun/project/X-Stream-open-source/inference/logs/%x_%j.out
#SBATCH -e /home/dyvm6xra/dyvm6xrauser04/peiwensun/project/X-Stream-open-source/inference/logs/%x_%j.err

set -euo pipefail

export XSTREAM_SPLIT=loose
export XSTREAM_MODE=cdpruner_token

bash /home/dyvm6xra/dyvm6xrauser04/peiwensun/project/X-Stream-open-source/inference/sbatch/run_v2_qwen3omni_original.sh
