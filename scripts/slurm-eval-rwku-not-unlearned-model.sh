#!/bin/bash -l
#SBATCH --job-name=slurm-eval-rwku-not-unlearned
#SBATCH --output=logs/slurm-eval-rwku-not-unlearned-%j.log
# Request the number of gpus using "--gres=gpu:<number>". E.g.:
#SBATCH --gres=gpu:2
# Request more time using "--time=<hours:mins:secs>". E.g.:
#SBATCH --time=01:30:00
# Request time partition "--partition=<Partition>". E.g.:
#SBATCH --partition=sc-gpu
set -euo pipefail

# Add host, time, and directory name for later troubleshooting
hostname; pwd; date

source "$HOME/anaconda3/etc/profile.d/conda.sh"
echo "Activate virtual environment (must exist)"
conda activate py312
echo "Run program in virtual environment"

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
cd "${REPO_DIR}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
MODEL_LABEL="${MODEL_LABEL:-Qwen2.5-1.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen2.5-1.5b-instruct/eval_rwku}"

BASE_OVERRIDES=(
  "evaluation.model_name_or_path=${MODEL_NAME_OR_PATH}"
  "evaluation.model.label=${MODEL_LABEL}"
  "evaluation.output_dir=${OUTPUT_DIR}"
  "evaluation.wandb.link_to_training_run=false"
  "evaluation.wandb.log_model_artifact=false"
)

echo "Evaluate non-unlearned model on RWKU using config/eval.yaml"
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Output directory: ${OUTPUT_DIR}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0}"
GPU_COUNT="${#GPU_IDS[@]}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Detected ${GPU_COUNT} visible GPU(s)"

if [[ "${GPU_COUNT}" -le 1 ]]; then
  python "${REPO_DIR}/eval/rwku/rwku.py" "${BASE_OVERRIDES[@]}" "$@"
else
  pids=()
  for SHARD_INDEX in "${!GPU_IDS[@]}"; do
    GPU_ID="${GPU_IDS[${SHARD_INDEX}]}"
    echo "Starting eval shard ${SHARD_INDEX}/${GPU_COUNT} on GPU ${GPU_ID}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${REPO_DIR}/eval/rwku/rwku.py" \
      "${BASE_OVERRIDES[@]}" \
      evaluation.shard.index="${SHARD_INDEX}" \
      evaluation.shard.count="${GPU_COUNT}" \
      "$@" &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  python "${REPO_DIR}/eval/rwku/merge_shards.py" \
    "${BASE_OVERRIDES[@]}" \
    evaluation.shard.index=0 \
    evaluation.shard.count="${GPU_COUNT}" \
    "$@"
fi

echo "Finished RWKU evaluation"
date
