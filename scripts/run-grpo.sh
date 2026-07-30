#!/bin/bash -l
#SBATCH --job-name=slurm-run-unlearning
#SBATCH --output=logs/slurm-run-unlearning-%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --qos=hopper
#SBATCH --partition=hopper

set -euo pipefail

STORAGE_DIR="${STORAGE_DIR:-/storage/scratch/lv13/lv13594}"
export HF_HOME="${HF_HOME:-${STORAGE_DIR}/huggingface}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HOME}/.cache/huggingface/token}"
CONDA_DIR="${CONDA_DIR:-${STORAGE_DIR}/anaconda3}"
TRAIN_ENV_PATH="${TRAIN_ENV_PATH:-${CONDA_DIR}/envs/py312_cu118}"
REPO_DIR="${REPO_DIR:-${STORAGE_DIR}/machine-unlearning-llm}"

hostname; pwd; date
source "${CONDA_DIR}/etc/profile.d/conda.sh"
echo "Activate virtual environment (must exist)"
conda activate "${TRAIN_ENV_PATH}"
echo "Run program in virtual environment"

cd "${REPO_DIR}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_DIR}/train.py}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_DIR}/config/train.yaml}"
SINGLE_GPU_CONFIG="${SINGLE_GPU_CONFIG:-${REPO_DIR}/config/accelerate_single_gpu.yaml}"
MULTI_GPU_CONFIG="${MULTI_GPU_CONFIG:-${REPO_DIR}/config/accelerate_multi_gpu.yaml}"
DEEPSPEED_ZERO3_CONFIG="${DEEPSPEED_ZERO3_CONFIG:-${REPO_DIR}/config/accelerate_deepspeed_zero3.yaml}"
RUN_NAME="${RUN_NAME:-unlearning-${SLURM_JOB_ID:-$$}}"
JOB_UNIQUE_ID="${SLURM_JOB_ID:-$$}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((20000 + JOB_UNIQUE_ID % 40000))}"
IFS=',' read -ra VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${NUM_GPUS:-${#VISIBLE_GPUS[@]}}"
if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "Invalid NUM_GPUS value: ${NUM_GPUS}" >&2
  exit 2
fi

PEFT_ENABLED="$(
  awk '
    $1 == "peft:" { in_peft=1; next }
    in_peft && $1 == "enabled:" { print tolower($2); exit }
    in_peft && /^[^[:space:]]/ { exit }
  ' "${TRAIN_CONFIG}"
)"
PEFT_ENABLED="${PEFT_ENABLED:-true}"
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
ACCELERATE_ARGS+=(--main_process_port "${MAIN_PROCESS_PORT}")

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "Detected ${NUM_GPUS} GPU(s)"
echo "PEFT enabled: ${PEFT_ENABLED}"
echo "Using accelerate config: ${ACCELERATE_CONFIG:-none}"
echo "Using main process port: ${MAIN_PROCESS_PORT}"
echo "Using accelerate args: ${ACCELERATE_ARGS[*]}"

nvidia-smi
nvidia-smi -q -d COMPUTE

python -c 'import torch; print("Hi"); print("count", torch.cuda.device_count()); torch.cuda.set_device(0); x=torch.ones(1, device="cuda"); print(x, torch.cuda.get_device_name(0))'

# echo "Run unlearning script"
accelerate launch \
  "${ACCELERATE_ARGS[@]}" \
  "${TRAIN_SCRIPT}" \
  "wandb.run_name=${RUN_NAME}" \
  "$@"
echo "Finished unlearning script"

# Add time for later troubleshooting
date
