#!/bin/bash -l
#SBATCH --job-name=analyze-completions
#SBATCH --output=logs/analyze-completions-%j.log
#SBATCH --time=01:30:00
#SBATCH --partition=hopper
set -euo pipefail
hostname; pwd; date

ENV_DIR="${ENV_DIR:-/storage/scratch/lv13/lv13594/}"
source "$ENV_DIR/anaconda3_bis/etc/profile.d/conda.sh"
conda activate py312_cu118_bis

cd "${REPO_DIR}"

# python eval/hold_out_styles/generate-and-analyze-completions.py \
#     concept='Nicolas Cage' \
#     model_name_or_path='Qwen/Qwen2.5-3B-Instruct' \

python eval/hold_out_styles/generate-and-analyze-completions.py \
    concept='Karl Marx' \
    model_name_or_path='./outputs/unlearning-sft-5259/checkpoint-17/' \

# 
# python eval/hold_out_styles/create_final_table.py
