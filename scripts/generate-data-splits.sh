#!/bin/bash -l
#SBATCH --job-name=data-splits
#SBATCH --output=logs/data-splits-%j.log
#SBATCH --time=01:00:00
set -euo pipefail

date

ENV_DIR="${ENV_DIR:-/storage/scratch/lv13/lv13594/}"
source "$ENV_DIR/anaconda3_bis/etc/profile.d/conda.sh"
conda activate py312_cu118_bis

python generate_sft_grpo_splits.py "$@"

date
