#!/bin/bash -l
#SBATCH --job-name=hold-out-eval
#SBATCH --output=logs/hold-out-eval-%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --partition=hopper
#SBATCH --qos=hopper
set -euo pipefail
hostname; pwd; date

ENV_DIR="${ENV_DIR:-/storage/scratch/lv13/lv13594/}"
REPO_DIR="${REPO_DIR:-/storage/scratch/lv13/lv13594/fresh-repo/grpo-llm-unlearning}"
source "$ENV_DIR/anaconda3_bis/etc/profile.d/conda.sh"
conda activate py312_cu118_bis

cd "${REPO_DIR}"

if [[ -n "${SKIP_IF_MARKER:-}" && -f "${SKIP_IF_MARKER}" ]]; then
  echo "Skipping hold-out evaluation because marker exists: ${SKIP_IF_MARKER}"
  cat "${SKIP_IF_MARKER}"
  exit 0
fi

python eval/behaviour/generate-and-analyze-completions.py "$@"
