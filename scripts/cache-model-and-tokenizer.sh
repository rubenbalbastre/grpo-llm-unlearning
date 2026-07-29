#!/bin/bash -l
#SBATCH --job-name=cache-model
#SBATCH --output=logs/cache-model-%j.log
#SBATCH --time=02:00:00
set -euo pipefail

STORAGE_DIR="${STORAGE_DIR:-/storage/scratch/lv13/lv13594}"
CONDA_DIR="${CONDA_DIR:-${STORAGE_DIR}/anaconda3}"
ENV_PATH="${ENV_PATH:-${CONDA_DIR}/envs/py312_cu118}"
REPO_DIR="${REPO_DIR:-${STORAGE_DIR}/machine-unlearning-llm}"

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"
cd "${REPO_DIR}"

python scripts/cache-model-and-tokenizer.py "$@"
