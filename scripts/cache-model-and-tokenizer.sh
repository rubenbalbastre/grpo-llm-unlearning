#!/bin/bash -l
#SBATCH --job-name=cache-model
#SBATCH --output=logs/cache-model-%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --partition=hopper
#SBATCH --qos=hopper
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/slurm-env.sh"

activate_training_env

python -m scripts.cache_model_and_tokenizer "$@"
