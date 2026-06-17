#!/bin/bash -l
#SBATCH --job-name=analyze-completions
#SBATCH --output=logs/analyze-completions-%j.log
#SBATCH --time=00:30:00
#SBATCH --partition=sc-cpu
set -euo pipefail

hostname; pwd; date
source "$HOME/anaconda3/etc/profile.d/conda.sh"
echo "Activate virtual environment (must exist)"
conda activate py312
echo "Run completion analysis"

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
cd "${REPO_DIR}"

CONCEPT="${CONCEPT:-Nicolas Cage}" # Karl Marx
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
FUZZY_THRESHOLD="${FUZZY_THRESHOLD:-0.84}"
FORGETTING_REWARD="${FORGETTING_REWARD:-1}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/completion_analysis/${MODEL_NAME//\//_}-${CONCEPT,,}-wandb-forgetting-reward-${FORGETTING_REWARD}.csv}"
MAX_EXAMPLES="${MAX_EXAMPLES:-40}"
WANDB_DOWNLOAD_DIR="${WANDB_DOWNLOAD_DIR:-outputs/completion_analysis/wandb-downloads}"

mkdir -p "$(dirname "${OUTPUT_CSV}")"

ARGS=()
ARGS+=("--wandb-download-dir" "${WANDB_DOWNLOAD_DIR}")
if [[ -n "${MODEL_NAME}" ]]; then
  ARGS+=("--model-name" "${MODEL_NAME}")
fi

echo "Downloading W&B completions for hydra.experiment.forget_concept=${CONCEPT}"
if [[ -n "${MODEL_NAME}" ]]; then
  echo "Filtering W&B runs for hydra.model.name=${MODEL_NAME}"
fi
echo "Output CSV: ${OUTPUT_CSV}"

python -u scripts/completitions_analysis/analyze-adaptive-completions.py \
  "${ARGS[@]}" \
  --concept "${CONCEPT}" \
  --near-threshold "${FUZZY_THRESHOLD}" \
  --forgetting-reward "${FORGETTING_REWARD}" \
  --max-examples "${MAX_EXAMPLES}" \
  --output-csv "${OUTPUT_CSV}"
