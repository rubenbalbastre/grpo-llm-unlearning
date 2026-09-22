#!/bin/bash -l
#SBATCH --job-name=data-splits
#SBATCH --output=logs/data-splits-%j.log
#SBATCH --time=01:00:00
set -euo pipefail

date

REPO_DIR="${REPO_DIR:-/storage/scratch/lv13/lv13594/fresh-repo/grpo-llm-unlearning}"
source "$REPO_DIR/anaconda3_bis/etc/profile.d/conda.sh"
conda activate py312_cu118_bis

python generate_sft_grpo_splits.py "$@"

date
