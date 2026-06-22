#!/bin/bash -l
#SBATCH --job-name=analyze-completions
#SBATCH --output=logs/analyze-completions-%j.log
#SBATCH --time=00:30:00
#SBATCH --partition=sc-cpu
set -euo pipefail
hostname; pwd; date

# SLURM
# source "$HOME/anaconda3/etc/profile.d/conda.sh"
# conda activate py312

# REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
# cd "${REPO_DIR}"

# LOCAL
source .venv/bin/activate


authors=(
  "Rihanna"
  "Karl Marx"
  "Confucius"
  "Nicolas Cage"
  "Justin Timberlake"
)

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
WANDB_DOWNLOAD_DIR="${WANDB_DOWNLOAD_DIR:-outputs/completion_analysis/wandb-downloads}"
REWARD_TYPE="${REWARD_TYPE:-r0}"

for CONCEPT in "${authors[@]}"; do

  python -u scripts/completitions_analysis/analyze-completions.py \
  --concept "${CONCEPT}" \
  "--model-name" "${MODEL_NAME}" \
  "--reward-type" "${REWARD_TYPE}" \
  "--wandb-download-dir" "${WANDB_DOWNLOAD_DIR}" \
  
done
