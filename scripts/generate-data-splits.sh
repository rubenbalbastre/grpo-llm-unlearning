#!/bin/bash -l
#SBATCH --job-name=data-splits
#SBATCH --output=logs/data-splits-%j.log
#SBATCH --time=01:00:00
set -euo pipefail

date

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate py312

python src/data_preprocessing/generate_sft_grpo_splits.py

date