#!/bin/bash -l
#SBATCH --job-name=cache-model
#SBATCH --output=logs/cache-model-%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --partition=hopper
#SBATCH --qos=hopper
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_ENV="${SCRIPT_DIR}/slurm-env.sh"
if [[ ! -f "${SLURM_ENV}" ]]; then
  SLURM_ENV="${REPO_DIR:-/storage/scratch/lv13/lv13594/machine-unlearning-llm}/scripts/slurm-env.sh"
fi
source "${SLURM_ENV}"

activate_training_env

python -m scripts.cache_model_and_tokenizer "$@"
