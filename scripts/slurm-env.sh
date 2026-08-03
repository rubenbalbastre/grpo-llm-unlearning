#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STORAGE_DIR="${STORAGE_DIR:-/storage/scratch/lv13/lv13594}"
CONDA_DIR="${CONDA_DIR:-${STORAGE_DIR}/anaconda3}"
TRAIN_ENV_PATH="${TRAIN_ENV_PATH:-${CONDA_DIR}/envs/py312_cu118}"
REPO_DIR="${REPO_DIR:-$(dirname "${SCRIPT_DIR}")}"

export HF_HOME="${HF_HOME:-${STORAGE_DIR}/huggingface}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HOME}/.cache/huggingface/token}"

activate_training_env() {
  source "${CONDA_DIR}/etc/profile.d/conda.sh"
  conda activate "${TRAIN_ENV_PATH}"
  cd "${REPO_DIR}"
}
