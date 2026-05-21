#!/bin/bash -l
#SBATCH --job-name=slurm-test-data-generator
#SBATCH --output=logs/slurm-test-data-generator-%j.log
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
echo "Run unlearning script"
# python /home/balalru/machine-unlearning-llm/test-data-generator.py
python /home/balalru/machine-unlearning-llm/test-safe-data-generator.py
date
