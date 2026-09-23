#!/bin/bash -l
#SBATCH --job-name=plot-training-diagnostics
#SBATCH --output=logs/plot-training-diagnostics-%j.log
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

set -euo pipefail

CONDA_DIR="${CONDA_DIR:-/storage/scratch/anaconda3_bis/}"
REPO_DIR="${REPO_DIR:-/storage/scratch/lv13/lv13594/fresh-repo/grpo-llm-unlearning}"

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate py312_cu118_bis
cd "${REPO_DIR}"

python scripts/figures/plot-training-diagnostics-by-model-size.py "$@"
