#!/bin/bash -l
#SBATCH --job-name=slurm-run-unlearning
#SBATCH --output=logs/slurm-run-unlearning-%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --partition=sc-gpu
set -euo pipefail

hostname; pwd; date
source "$HOME/anaconda3/etc/profile.d/conda.sh"
echo "Activate virtual environment (must exist)"
conda activate py312
echo "Run program in virtual environment"

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_DIR}/train.py}"
SINGLE_GPU_CONFIG="${SINGLE_GPU_CONFIG:-${REPO_DIR}/configs/accelerate_single_gpu.yaml}"
MULTI_GPU_CONFIG="${MULTI_GPU_CONFIG:-${REPO_DIR}/configs/accelerate_multi_gpu.yaml}"
IFS=',' read -ra VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${NUM_GPUS:-${#VISIBLE_GPUS[@]}}"
if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "Invalid NUM_GPUS value: ${NUM_GPUS}" >&2
  exit 2
fi

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${MULTI_GPU_CONFIG}}"
else
  ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${SINGLE_GPU_CONFIG}}"
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "Detected ${NUM_GPUS} GPU(s)"
echo "Using accelerate config: ${ACCELERATE_CONFIG}"

echo "Run unlearning script"
accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_GPUS}" \
  "${TRAIN_SCRIPT}" \
  "$@"
echo "Finished unlearning script"

echo "Evaluate trained model on RWKU using configs/eval.yaml"
python "${REPO_DIR}/eval/rwku/rwku.py"
echo "Finished RWKU evaluation"

# Add time for later troubleshooting
date
