#!/bin/bash -l
#SBATCH --job-name=scratch-inspect-data-generator
#SBATCH --output=logs/scratch-inspect-data-generator-%j.log
# Request the number of gpus usint "--gres=gpu:<number>". E.g.:
#SBATCH --gres=gpu:1
# Request more time using "--time=<hours:mins:secs>". E.g.:
#SBATCH --time=00:05:00
# Request time partition "--partition=<Partition>". E.g.:
#SBATCH --partition=sc-gpu
# Add host, time, and directory name for later troubleshooting
hostname; pwd; date
# Run the program/command
source "$HOME/anaconda3/etc/profile.d/conda.sh"
echo "Activate virtual environment (must exist)"
conda activate py312
echo "Run program in virtual environment"
REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
cd "${REPO_DIR}"
echo "Run data generator inspection"
python -m scratch.inspect_data_generator
date
