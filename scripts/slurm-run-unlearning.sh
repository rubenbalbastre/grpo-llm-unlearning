#!/bin/bash -l
#SBATCH --job-name=slurm-run-unlearning
#SBATCH --output=logs/slurm-run-unlearning-%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=sc-gpu
set -euo pipefail

hostname; pwd; date
source "$HOME/anaconda3/etc/profile.d/conda.sh"
echo "Activate virtual environment (must exist)"
conda activate py312
echo "Run program in virtual environment"

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_DIR}/train_adaptive_grpo.py}"
SINGLE_GPU_CONFIG="${SINGLE_GPU_CONFIG:-${REPO_DIR}/configs/accelerate_single_gpu.yaml}"
MULTI_GPU_CONFIG="${MULTI_GPU_CONFIG:-${REPO_DIR}/configs/accelerate_multi_gpu.yaml}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "Invalid NUM_GPUS value: ${NUM_GPUS}" >&2
  exit 2
fi

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${MULTI_GPU_CONFIG}}"
else
  ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${SINGLE_GPU_CONFIG}}"
fi

echo "Detected ${NUM_GPUS} GPU(s)"
echo "Using accelerate config: ${ACCELERATE_CONFIG}"

echo "Run unlearning script"
accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_GPUS}" \
  "${TRAIN_SCRIPT}" \
  "$@"
echo "Finished unlearning script"

# evaluate RWKU
# echo "Evaluate RWKU"
# python /home/balalru/machine-unlearning-llm/eval/rwku/rwku.py \
#     --model_name_or_path /home/balalru/machine-unlearning-llm/outputs/adaptive_grpo_min/final_model \
#     --output_dir /home/balalru/machine-unlearning-llm/outputs/eval_rwku/stephen_king \
#     --subjects "Stephen King"
# Add time for later troubleshooting
date
