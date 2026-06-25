#!/bin/bash -l
#SBATCH --job-name=compare-rwku-generations
#SBATCH --output=logs/compare-rwku-generations-%j.log
#SBATCH --time=00:10:00
#SBATCH --partition=sc-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
set -euo pipefail

hostname; pwd; date
source "$HOME/anaconda3/etc/profile.d/conda.sh"
echo "Activate virtual environment (must exist)"
conda activate py312
echo "Run RWKU generation comparison"

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
cd "${REPO_DIR}"

LEFT_JSONL="${LEFT_JSONL:-outputs/unlearning-4261/checkpoint-159/eval_rwku/rwku_generations.jsonl}"
RIGHT_JSONL="${RIGHT_JSONL:-outputs/unlearning-4260/checkpoint-159/eval_rwku/rwku_generations.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rwku_generation_diffs/unlearning-4261_vs_unlearning-4260_checkpoint-159}"

echo "Left JSONL: ${LEFT_JSONL}"
echo "Right JSONL: ${RIGHT_JSONL}"
echo "Output directory: ${OUTPUT_DIR}"

python -u scripts/compare-rwku-generations.py \
  --left "${LEFT_JSONL}" \
  --right "${RIGHT_JSONL}" \
  --output-dir "${OUTPUT_DIR}"

echo "Finished RWKU generation comparison"
date
