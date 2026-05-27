#!/bin/bash -l
#SBATCH --job-name=slurm-eval-rwku
#SBATCH --output=logs/slurm-eval-rwku-%j.log
# Request the number of gpus usint "--gres=gpu:<number>". E.g.:
#SBATCH --gres=gpu:1
# Request more time using "--time=<hours:mins:secs>". E.g.:
#SBATCH --time=01:30:00
# Request time partition "--partition=<Partition>". E.g.:
#SBATCH --partition=sc-gpu
set -euo pipefail
# Add host, time, and directory name for later troubleshooting
hostname; pwd; date
# Run the program/command
source "$HOME/anaconda3/etc/profile.d/conda.sh"
echo "Activate virtual environment (must exist)"
conda activate py312
echo "Run program in virtual environment"

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
cd "${REPO_DIR}"
echo "Evaluate RWKU using config/eval.yaml"
python "${REPO_DIR}/eval/rwku/rwku.py"
echo "Finished RWKU evaluation"
# Add time for later troubleshooting
date
