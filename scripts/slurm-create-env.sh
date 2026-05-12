#!/bin/bash -l
#SBATCH --job-name=slurm-create-env
#SBATCH --output=logs/slurm-create-env-%j.log
# Request the number of gpus usint "--gres=gpu:<number>". E.g.:
#SBATCH --gres=gpu:1
# Request more time using "--time=<hours:mins:secs>". E.g.:
#SBATCH --time=00:30:00
# Request time partition "--partition=<Partition>". E.g.:
#SBATCH --partition=sc-gpu
# Add host, time, and directory name for later troubleshooting
hostname; pwd; date
# Run the program/command
echo "Create virtual environment"
python3 -m venv /home/balalru/machine-unlearning-llm/venv
echo "Activate virtual environment"
source /home/balalru/machine-unlearning-llm/venv/bin/activate
echo "Install requirements (file must contain required packages)"
pip install -r /home/balalru/machine-unlearning-llm/requirements.txt
# Add time for later troubleshooting
date