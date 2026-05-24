#!/bin/bash
#SBATCH -J STRM-peiwen-eval          # 任务名
#SBATCH -N 1                         # 申请 1 个节点
#SBATCH --gres=gpu:4                 # 每个节点申请 1 张 GPU
#SBATCH --cpus-per-gpu=16            # 每张 GPU 对应申请 16 CPU 核心
#SBATCH --mem-per-cpu=14G            # 每个 CPU 核心对应申请 14G 内存
#SBATCH -o logs/%x_%j/%x_%j.out      # stdout 日志：任务名_任务ID.out
#SBATCH -e logs/%x_%j/%x_%j.err      # stderr 日志：任务名_任务ID.err

set -euo pipefail

echo "JobID: ${SLURM_JOB_ID}"
echo "NodeList: ${SLURM_NODELIST}"
echo "CONDA_EXE: ${CONDA_EXE}"

# sbatch -A dyvm6xra -p q-dyvm6xra -q qos_g5_24hrs -t 24:00:00
# conda 环境
# CONDA_BASE="$("$CONDA_EXE" info --base)"
# source "$CONDA_BASE/etc/profile.d/conda.sh"
# conda activate myenv

cd /home/dyvm6xra/dyvm6xrauser04/peiwensun/project/VLLMFlow/projects/

# 可选：打印 GPU 状态
nvidia-smi

bash run_double_copy.sh --ckpt-list-file /home/dyvm6xra/dyvm6xrauser04/peiwensun/project/LLaMA-Factory_v0122/saves/qwen3.5-9b/full/sft_test_xstream_1e_5_code_20260427/ckpt.txt

echo "Done."
