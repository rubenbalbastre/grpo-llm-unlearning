#!/bin/bash -l
#SBATCH --job-name=data-splits
#SBATCH --output=logs/data-splits-%j.log
#SBATCH --time=01:00:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_ENV="${SCRIPT_DIR}/slurm-env.sh"
if [[ ! -f "${SLURM_ENV}" ]]; then
  SLURM_ENV="${REPO_DIR:-/storage/scratch/lv13/lv13594/machine-unlearning-llm}/scripts/slurm-env.sh"
fi
source "${SLURM_ENV}"

date

activate_training_env

python generate_sft_grpo_splits.py "$@"

date
