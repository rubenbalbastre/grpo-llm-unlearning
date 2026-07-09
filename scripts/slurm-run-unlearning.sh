#!/bin/bash -l
#SBATCH --job-name=slurm-run-unlearning
#SBATCH --output=logs/slurm-run-unlearning-%j.log
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --partition=sc-gpu
set -euo pipefail

hostname; pwd; date
source "$HOME/anaconda3/etc/profile.d/conda.sh"
echo "Activate virtual environment (must exist)"
conda activate py312
echo "Run program in virtual environment"

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
cd "${REPO_DIR}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_DIR}/train.py}"
SINGLE_GPU_CONFIG="${SINGLE_GPU_CONFIG:-${REPO_DIR}/config/accelerate_single_gpu.yaml}"
MULTI_GPU_CONFIG="${MULTI_GPU_CONFIG:-${REPO_DIR}/config/accelerate_multi_gpu.yaml}"
DEEPSPEED_ZERO3_CONFIG="${DEEPSPEED_ZERO3_CONFIG:-${REPO_DIR}/config/accelerate_deepspeed_zero3.yaml}"
RUN_NAME="${RUN_NAME:-unlearning-${SLURM_JOB_ID:-$$}}"
IFS=',' read -ra VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${NUM_GPUS:-${#VISIBLE_GPUS[@]}}"
if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "Invalid NUM_GPUS value: ${NUM_GPUS}" >&2
  exit 2
fi

PEFT_ENABLED="true"
for arg in "$@"; do
  case "${arg}" in
    peft.enabled=false|peft.enabled=False|peft.enabled=0)
      PEFT_ENABLED="false"
      ;;
    peft.enabled=true|peft.enabled=True|peft.enabled=1)
      PEFT_ENABLED="true"
      ;;
  esac
done

if [[ -n "${ACCELERATE_CONFIG:-}" ]]; then
  ACCELERATE_ARGS=(--config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_GPUS}")
elif [[ "${NUM_GPUS}" -gt 1 && "${PEFT_ENABLED}" == "false" ]]; then
  ACCELERATE_CONFIG="${DEEPSPEED_ZERO3_CONFIG}"
  ACCELERATE_ARGS=(--config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_GPUS}")
elif [[ "${NUM_GPUS}" -gt 1 ]]; then
  ACCELERATE_CONFIG="${MULTI_GPU_CONFIG}"
  ACCELERATE_ARGS=(--config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_GPUS}")
else
  ACCELERATE_CONFIG="${SINGLE_GPU_CONFIG}"
  ACCELERATE_ARGS=(--config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_GPUS}")
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "Detected ${NUM_GPUS} GPU(s)"
echo "PEFT enabled: ${PEFT_ENABLED}"
echo "Using accelerate config: ${ACCELERATE_CONFIG:-none}"
echo "Using accelerate args: ${ACCELERATE_ARGS[*]}"

# echo "Run unlearning script"
accelerate launch \
  "${ACCELERATE_ARGS[@]}" \
  "${TRAIN_SCRIPT}" \
  "wandb.run_name=${RUN_NAME}" \
  "$@"
echo "Finished unlearning script"

# Add time for later troubleshooting
date
