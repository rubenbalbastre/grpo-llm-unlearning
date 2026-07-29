#!/bin/bash -l
#SBATCH --job-name=data-splits
#SBATCH --output=logs/data-splits-%j.log
#SBATCH --time=01:00:00
set -euo pipefail

STORAGE_DIR="${STORAGE_DIR:-/storage/scratch/lv13/lv13594}"
CONDA_DIR="${CONDA_DIR:-${STORAGE_DIR}/anaconda3}"
TRAIN_ENV_PATH="${TRAIN_ENV_PATH:-${CONDA_DIR}/envs/py312_cu118}"
REPO_DIR="${REPO_DIR:-${STORAGE_DIR}/machine-unlearning-llm}"

date

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${TRAIN_ENV_PATH}"
cd "${REPO_DIR}"

python generate_sft_grpo_splits.py "$@"

date
