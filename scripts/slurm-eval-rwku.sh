#!/bin/bash -l
#SBATCH --job-name=slurm-eval-rwku
#SBATCH --output=logs/slurm-eval-rwku-%j.log
# Request the number of gpus usint "--gres=gpu:<number>". E.g.:
#SBATCH --gres=gpu:2
# Request more time using "--time=<hours:mins:secs>". E.g.:
#SBATCH --time=01:30:00
# Request time partition "--partition=<Partition>". E.g.:
#SBATCH --partition=sc-gpu
set -euo pipefail
# Add host, time, and directory name for later troubleshooting
hostname; pwd; date
# Run the program/command
source "$HOME/anaconda3/etc/profile.d/conda.sh"
echo "Activate virtual environment (must exist)"
conda activate py312
echo "Run program in virtual environment"

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
cd "${REPO_DIR}"
echo "Evaluate RWKU using config/eval.yaml"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0}"
GPU_COUNT="${#GPU_IDS[@]}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Detected ${GPU_COUNT} visible GPU(s)"

if [[ "${GPU_COUNT}" -le 1 ]]; then
  python "${REPO_DIR}/eval/rwku/rwku.py"
else
  pids=()
  for SHARD_INDEX in "${!GPU_IDS[@]}"; do
    GPU_ID="${GPU_IDS[${SHARD_INDEX}]}"
    echo "Starting eval shard ${SHARD_INDEX}/${GPU_COUNT} on GPU ${GPU_ID}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${REPO_DIR}/eval/rwku/rwku.py" \
      evaluation.shard.index="${SHARD_INDEX}" \
      evaluation.shard.count="${GPU_COUNT}" &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  python "${REPO_DIR}/eval/rwku/merge_shards.py" \
    evaluation.shard.index=0 \
    evaluation.shard.count="${GPU_COUNT}"
fi
echo "Finished RWKU evaluation"
# Add time for later troubleshooting
date
