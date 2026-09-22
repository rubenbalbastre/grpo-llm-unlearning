#!/bin/bash -l
#SBATCH --job-name=sft
#SBATCH --output=logs/sft-%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=hopper
set -euo pipefail

date
ENV_DIR="${ENV_DIR:-/storage/scratch/lv13/lv13594/}"
source "$ENV_DIR/anaconda3_bis/etc/profile.d/conda.sh"
conda activate py312_cu118_bis

python sft_warm_up.py "$@"

date