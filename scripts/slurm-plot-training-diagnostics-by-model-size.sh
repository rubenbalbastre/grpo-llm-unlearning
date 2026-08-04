#!/bin/bash -l
#SBATCH --job-name=plot-training-diagnostics
#SBATCH --output=logs/plot-training-diagnostics-%j.log
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

set -euo pipefail

CONDA_DIR="${CONDA_DIR:-/home/balalru/anaconda3}"
TRAIN_ENV_PATH="${TRAIN_ENV_PATH:-${CONDA_DIR}}"
REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate py312
cd "${REPO_DIR}"

python scripts/figures/plot-training-diagnostics-by-model-size.py "$@"
