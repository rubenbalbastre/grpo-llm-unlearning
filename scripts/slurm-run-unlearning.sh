#!/bin/bash -l
#SBATCH --job-name=slurm-run-env-gpu
#SBATCH --output=logs/slurm-run-env-gpu-%j.log
# Request the number of gpus usint "--gres=gpu:<number>". E.g.:
#SBATCH --gres=gpu:2
# Request more time using "--time=<hours:mins:secs>". E.g.:
#SBATCH --time=00:30:00
# Request time partition "--partition=<Partition>". E.g.:
#SBATCH --partition=sc-gpu# Add host, time, and directory name for later troubleshooting
hostname; pwd; date
# Run the program/command
source "$HOME/anaconda3/etc/profile.d/conda.sh"
echo "Activate virtual environment (must exist)"
conda activate py312
echo "Run program in virtual environment"
python /home/balalru/machine-unlearning-llm/llm-attacker.py
# python /home/balalru/machine-unlearning-llm/train_adaptive_grpo.py
# Add time for later troubleshooting
date