#!/bin/bash -l
#SBATCH --job-name=slurm-run-unlearning
#SBATCH --output=logs/slurm-run-unlearning-%j.log
#SBATCH --gres=gpu:2
#SBATCH --time=03:00:00
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
RUN_NAME="${RUN_NAME:-unlearning-${SLURM_JOB_ID:-$$}}"
OUTPUT_DIR="${REPO_DIR}/outputs/${RUN_NAME}"
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

# echo "Run unlearning script"
accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_GPUS}" \
  "${TRAIN_SCRIPT}" \
  "wandb.run_name=${RUN_NAME}" \
  "$@"
echo "Finished unlearning script"

EVAL_MODEL_DIR="${OUTPUT_DIR}/final_model"
EVAL_OUTPUT_DIR="${OUTPUT_DIR}/eval_rwku"
if [[ ! -d "${EVAL_MODEL_DIR}" ]]; then
  echo "Expected final model directory does not exist: ${EVAL_MODEL_DIR}" >&2
  exit 1
fi
EVAL_OVERRIDES=(
  "evaluation.model_name_or_path=${EVAL_MODEL_DIR}"
  "evaluation.output_dir=${EVAL_OUTPUT_DIR}"
)
echo "Evaluate trained model on RWKU using config/eval.yaml"
if [[ "${NUM_GPUS}" -le 1 ]]; then
  python "${REPO_DIR}/eval/rwku/rwku.py" "${EVAL_OVERRIDES[@]}"
else
  pids=()
  for SHARD_INDEX in "${!VISIBLE_GPUS[@]}"; do
    GPU_ID="${VISIBLE_GPUS[${SHARD_INDEX}]}"
    echo "Starting eval shard ${SHARD_INDEX}/${NUM_GPUS} on GPU ${GPU_ID}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${REPO_DIR}/eval/rwku/rwku.py" \
      "${EVAL_OVERRIDES[@]}" \
      evaluation.shard.index="${SHARD_INDEX}" \
      evaluation.shard.count="${NUM_GPUS}" &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  python "${REPO_DIR}/eval/rwku/merge_shards.py" \
    "${EVAL_OVERRIDES[@]}" \
    evaluation.shard.index=0 \
    evaluation.shard.count="${NUM_GPUS}"
fi
echo "Finished RWKU evaluation"

# Add time for later troubleshooting
date
