#!/bin/bash -l
#SBATCH --job-name=analyze-completions
#SBATCH --output=logs/analyze-completions-%j.log
#SBATCH --time=01:30:00
#SBATCH --partition=sc-gpu
set -euo pipefail
hostname; pwd; date

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate py312

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
cd "${REPO_DIR}"

python scripts/completions_analysis/generate-and-analyze-completions.py \
    concept='Nicolas Cage' \
    model_name_or_path='Qwen/Qwen2.5-3B-Instruct' \
    dataset_path='data/hold_out_nicolas_cage' \
    output_dir='outputs/completion_analysis/generated-completions-nicolas-cage-unlearning-3B-mini'

python scripts/completions_analysis/generate-and-analyze-completions.py \
    concept='Nicolas Cage' \
    model_name_or_path='./outputs/unlearning-sft-5104/checkpoint-3/' \
    dataset_path='data/hold_out_nicolas_cage' \
    output_dir='outputs/completion_analysis/generated-completions-nicolas-cage-unlearning-5104-mini'

# 
# python scripts/completions_analysis/create_final_table.py
