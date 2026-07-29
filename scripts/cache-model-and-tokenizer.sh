#!/bin/bash -l
#SBATCH --job-name=cache-model
#SBATCH --output=logs/cache-model-%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --partition=hopper
#SBATCH --qos=hopper
set -euo pipefail

STORAGE_DIR="${STORAGE_DIR:-/storage/scratch/lv13/lv13594}"
CONDA_DIR="${CONDA_DIR:-${STORAGE_DIR}/anaconda3}"
TRAIN_ENV_PATH="${TRAIN_ENV_PATH:-${CONDA_DIR}/envs/py312_cu118}"
REPO_DIR="${REPO_DIR:-${STORAGE_DIR}/machine-unlearning-llm}"

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${TRAIN_ENV_PATH}"
cd "${REPO_DIR}"

python -m scripts.cache_model_and_tokenizer "$@"
