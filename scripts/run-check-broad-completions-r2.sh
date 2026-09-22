#!/bin/bash -l
#SBATCH --job-name=check-broad-completions
#SBATCH --output=logs/check-broad-completions-%j.log
#SBATCH --time=00:20:00
set -euo pipefail
hostname; pwd; date

ENV_DIR="${ENV_DIR:-/storage/scratch/lv13/lv13594/}"
source "$ENV_DIR/anaconda3_bis/etc/profile.d/conda.sh"
conda activate py312_cu118_bis

cd "${REPO_DIR}"

python3 scripts/check-broad-completions-r2.py
