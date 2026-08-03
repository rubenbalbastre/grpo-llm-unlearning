#!/bin/bash -l
#SBATCH --job-name=analyze-completions
#SBATCH --output=logs/analyze-completions-%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --partition=sc-gpu
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_ENV="${SCRIPT_DIR}/slurm-env.sh"
if [[ ! -f "${SLURM_ENV}" ]]; then
  SLURM_ENV="${REPO_DIR:-/storage/scratch/lv13/lv13594/machine-unlearning-llm}/scripts/slurm-env.sh"
fi
source "${SLURM_ENV}"

hostname; pwd; date
activate_training_env

if [[ -n "${SKIP_IF_MARKER:-}" && -f "${SKIP_IF_MARKER}" ]]; then
  echo "Skipping hold-out evaluation because marker exists: ${SKIP_IF_MARKER}"
  cat "${SKIP_IF_MARKER}"
  exit 0
fi

python eval/hold_out_styles/generate-and-analyze-completions.py "$@"
