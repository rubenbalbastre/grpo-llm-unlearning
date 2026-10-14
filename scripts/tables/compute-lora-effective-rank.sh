#!/bin/bash -l
#SBATCH --job-name=lora-effective-rank
#SBATCH --output=logs/lora-effective-rank-%j.log
#SBATCH --time=01:00:00
#SBATCH --mem=16G
set -euo pipefail

ENV_DIR="${ENV_DIR:-/storage/scratch/lv13/lv13594/}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

echo "${REPO_DIR}"

source "$ENV_DIR/anaconda3_bis/etc/profile.d/conda.sh"
conda activate py312_cu118_bis

cd "$REPO_DIR"
python3 scripts/tables/compute-lora-effective-rank.py "$@"
