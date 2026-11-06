#!/bin/bash -l
#SBATCH --job-name=compress-outputs
#SBATCH --output=logs/compress-outputs-%j.log
#SBATCH --time=04:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --partition=hopper
#SBATCH --qos=hopper
set -euo pipefail

REPO_DIR="${REPO_DIR:-/storage/scratch/lv13/lv13594/fresh-repo/grpo-llm-unlearning}"

cd "$REPO_DIR"
test -d outputs
tar -czvf outputs.tar.gz outputs/
