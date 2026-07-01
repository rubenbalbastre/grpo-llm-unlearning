#!/bin/bash -l
#SBATCH --job-name=analyze-completions
#SBATCH --output=logs/analyze-completions-%j.log
#SBATCH --time=00:30:00
#SBATCH --partition=sc-gpu
set -euo pipefail
hostname; pwd; date

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate py312

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
cd "${REPO_DIR}"

python scripts/completions_analysis/create_final_table.py

# python scripts/create_hold_out_dataset.py
# python scripts/completions_analysis/generate-and-analyze-completions.py

# python scripts/completions_analysis/create_hold_out_dataset.py --concept='Rihanna'
# python scripts/completions_analysis/generate-and-analyze-completions.py --concept='Rihanna' --model-name-or-path='./outputs/unlearning-4548/checkpoint-159' --dataset-path='data/hold_out_rihanna' --output-dir="outputs/completion_analysis/generated-completions-rihanna-unlearning-4548"
# python scripts/completions_analysis/generate-and-analyze-completions.py --concept='Rihanna' --model-name-or-path='./outputs/unlearning-4550/checkpoint-159' --dataset-path='data/hold_out_rihanna' --output-dir="outputs/completion_analysis/generated-completions-rihanna-unlearning-4550"
# python scripts/completions_analysis/generate-and-analyze-completions.py --concept='Rihanna' --model-name-or-path='./outputs/unlearning-4563/checkpoint-159' --dataset-path='data/hold_out_rihanna' --output-dir="outputs/completion_analysis/generated-completions-rihanna-unlearning-4563"

# python scripts/completions_analysis/create_hold_out_dataset.py --concept='Nicolas Cage'
# python scripts/completions_analysis/generate-and-analyze-completions.py --concept='Nicolas Cage' --model-name-or-path='./outputs/unlearning-4557/checkpoint-159' --dataset-path='data/hold_out_nicolas_cage' --output-dir="outputs/completion_analysis/generated-completions-nicolas-cage-unlearning-4557"
# python scripts/completions_analysis/generate-and-analyze-completions.py --concept='Nicolas Cage' --model-name-or-path='./outputs/unlearning-4559/checkpoint-159' --dataset-path='data/hold_out_nicolas_cage' --output-dir="outputs/completion_analysis/generated-completions-nicolas-cage-unlearning-4559"
# python scripts/completions_analysis/generate-and-analyze-completions.py --concept='Nicolas Cage' --model-name-or-path='./outputs/unlearning-4558/checkpoint-159' --dataset-path='data/hold_out_nicolas_cage' --output-dir="outputs/completion_analysis/generated-completions-nicolas-cage-unlearning-4558"

# python scripts/completions_analysis/create_hold_out_dataset.py --concept='Karl Marx'
# python scripts/completions_analysis/generate-and-analyze-completions.py --concept='Karl Marx' --model-name-or-path='./outputs/unlearning-4551/checkpoint-159' --dataset-path='data/hold_out_karl_marx' --output-dir="outputs/completion_analysis/generated-completions-karl-marx-unlearning-4551"
# python scripts/completions_analysis/generate-and-analyze-completions.py --concept='Karl Marx' --model-name-or-path='./outputs/unlearning-4553/checkpoint-159' --dataset-path='data/hold_out_karl_marx' --output-dir="outputs/completion_analysis/generated-completions-karl-marx-unlearning-4553"
# python scripts/completions_analysis/generate-and-analyze-completions.py --concept='Karl Marx' --model-name-or-path='./outputs/unlearning-4552/checkpoint-159' --dataset-path='data/hold_out_karl_marx' --output-dir="outputs/completion_analysis/generated-completions-karl-marx-unlearning-4552"

# python scripts/completions_analysis/create_hold_out_dataset.py --concept='Justin Timberlake'
# python scripts/completions_analysis/generate-and-analyze-completions.py --concept='Justin Timberlake' --model-name-or-path='./outputs/unlearning-4560/checkpoint-159' --dataset-path='data/hold_out_justin_timberlake' --output-dir="outputs/completion_analysis/generated-completions-justin-timberlake-unlearning-4560"
# python scripts/completions_analysis/generate-and-analyze-completions.py --concept='Justin Timberlake' --model-name-or-path='./outputs/unlearning-4561/checkpoint-159' --dataset-path='data/hold_out_justin_timberlake' --output-dir="outputs/completion_analysis/generated-completions-justin-timberlake-unlearning-4561"
# python scripts/completions_analysis/generate-and-analyze-completions.py --concept='Justin Timberlake' --model-name-or-path='./outputs/unlearning-4562/checkpoint-159' --dataset-path='data/hold_out_justin_timberlake' --output-dir="outputs/completion_analysis/generated-completions-justin-timberlake-unlearning-4562"

# python scripts/completions_analysis/create_hold_out_dataset.py --concept='Confucius'
# python scripts/completions_analysis/generate-and-analyze-completions.py --concept='Confucius' --model-name-or-path='./outputs/unlearning-4556/checkpoint-159' --dataset-path='data/hold_out_confucius' --output-dir="outputs/completion_analysis/generated-completions-confucius-unlearning-4556"
# python scripts/completions_analysis/generate-and-analyze-completions.py --concept='Confucius' --model-name-or-path='./outputs/unlearning-4554/checkpoint-159' --dataset-path='data/hold_out_confucius' --output-dir="outputs/completion_analysis/generated-completions-confucius-unlearning-4554"
# python scripts/completions_analysis/generate-and-analyze-completions.py --concept='Confucius' --model-name-or-path='./outputs/unlearning-4555/checkpoint-159' --dataset-path='data/hold_out_confucius' --output-dir="outputs/completion_analysis/generated-completions-confucius-unlearning-4555"