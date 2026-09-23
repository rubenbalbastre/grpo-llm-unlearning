#!/bin/bash -l
#SBATCH --job-name=sft
#SBATCH --output=logs/sft-%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=hopper
#SBATCH --qos=hopper
set -euo pipefail

date
ENV_DIR="${ENV_DIR:-/storage/scratch/lv13/lv13594/}"
REPO_DIR="${REPO_DIR:-/storage/scratch/lv13/lv13594/fresh-repo/grpo-llm-unlearning}"
source "$ENV_DIR/anaconda3_bis/etc/profile.d/conda.sh"
conda activate py312_cu118_bis
cd "${REPO_DIR}"

JOB_UNIQUE_ID="${SLURM_JOB_ID:-$$}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((20000 + JOB_UNIQUE_ID % 40000))}"

accelerate launch \
    --config_file config/accelerate_single_gpu.yaml \
    --num_processes 1 \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    sft_warm_up.py \
    "$@"

date
