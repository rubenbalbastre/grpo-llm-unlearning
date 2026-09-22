#!/bin/bash -l
#SBATCH --job-name=check-broad-completions
#SBATCH --output=logs/check-broad-completions-%j.log
#SBATCH --time=00:20:00
set -euo pipefail
hostname; pwd; date

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate py312

REPO_DIR="${REPO_DIR:-/storage/scratch/lv13/lv13594/fresh-repo/grpo-llm-unlearning}"
cd "${REPO_DIR}"

python3 scripts/check-broad-completions-r2.py
