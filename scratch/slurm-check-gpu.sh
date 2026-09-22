#!/bin/bash -l
#SBATCH --job-name=scratch-check-gpu
#SBATCH --output=logs/scratch-check-gpu-%j.log
# Request the number of gpus usint "--gres=gpu:<number>". E.g.:
#SBATCH --gres=gpu:2
# Request more time using "--time=<hours:mins:secs>". E.g.:
#SBATCH --time=00:30:00
# Request time partition "--partition=<Partition>". E.g.:
#SBATCH --partition=hopper# Add host, time, and directory name for later troubleshooting
hostname; pwd; date
# Run the program/command
REPO_DIR="${REPO_DIR:-/storage/scratch/lv13/lv13594/fresh-repo/grpo-llm-unlearning}"
source "$REPO_DIR/anaconda3_bis/etc/profile.d/conda.sh"
echo "Activate virtual environment (must exist)"
conda activate py312_cu118_bis
echo "Run program in virtual environment"
REPO_DIR="${REPO_DIR:-/storage/scratch/lv13/lv13594/fresh-repo/grpo-llm-unlearning}"
cd "${REPO_DIR}"
python -m scratch.check_gpu
# Add time for later troubleshooting
date
