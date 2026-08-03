#!/bin/bash -l
#SBATCH --job-name=check-broad-completions
#SBATCH --output=logs/check-broad-completions-%j.log
#SBATCH --time=00:20:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_ENV="${SCRIPT_DIR}/slurm-env.sh"
if [[ ! -f "${SLURM_ENV}" ]]; then
  SLURM_ENV="${REPO_DIR:-/storage/scratch/lv13/lv13594/machine-unlearning-llm}/scripts/slurm-env.sh"
fi
source "${SLURM_ENV}"

hostname; pwd; date
activate_training_env

python3 scripts/check-broad-completions-r2.py
