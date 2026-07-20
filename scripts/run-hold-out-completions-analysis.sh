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

# python eval/hold_out_styles/generate-and-analyze-completions.py \
#     concept='Nicolas Cage' \
#     model_name_or_path='Qwen/Qwen2.5-3B-Instruct' \

python eval/hold_out_styles/generate-and-analyze-completions.py \
    concept='Nicolas Cage' \
    model_name_or_path='./outputs/unlearning-sft-5225/checkpoint-14/' \

# 
# python eval/hold_out_styles/create_final_table.py
