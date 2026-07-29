#!/bin/bash -l
#SBATCH --job-name=check-broad-completions
#SBATCH --output=logs/check-broad-completions-%j.log
#SBATCH --time=00:20:00
set -euo pipefail
hostname; pwd; date

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate py312_cu118

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
cd "${REPO_DIR}"

python3 scripts/check-broad-completions-r2.py
