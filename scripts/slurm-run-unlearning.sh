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
echo "Activate virtual environment (must exist)"
source /home/balalru/machine-unlearning-llm/venv/bin/activate
echo "Run program in virtual environment"
python3 /home/balalru/machine-unlearning-llm/test-gpu.py
# Add time for later troubleshooting
date