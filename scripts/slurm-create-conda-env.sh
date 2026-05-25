#!/bin/bash -l
#SBATCH --job-name=slurm-create-conda-env
#SBATCH --output=logs/slurm-create-conda-env-%j.log
# Request the number of gpus usint "--gres=gpu:<number>". E.g.:
#SBATCH --gres=gpu:1
# Request more time using "--time=<hours:mins:secs>". E.g.:
#SBATCH --time=00:30:00
# Request time partition "--partition=<Partition>". E.g.:
#SBATCH --partition=sc-gpu

# Add host, time, and directory name for later troubleshooting
hostname; pwd; date

source "$HOME/anaconda3/etc/profile.d/conda.sh"
ENV_PATH="$HOME/anaconda3/envs/py312"

echo "Checking for environment at $ENV_PATH"
if ! conda env list | awk '{print $1}' | grep -qx "py312"; then
  echo "Create environment at $ENV_PATH"
  conda create -y -p "$ENV_PATH" python=3.12.3
else
  echo "Environment already exists at $ENV_PATH"
fi

echo "Activate environment"
conda activate "$ENV_PATH"

echo "Installing high-speed dependency manager"
pip install --upgrade uv

echo "Executing proper multi-repository installation"

# 1. Clear out any broken build artifacts from the last failure
uv cache clean
rm -rf ~/.cache/uv/

# 2. Run the unified installation explicitly locking onto pre-built cu126 binaries
uv pip install \
  --extra-index-url https://pytorch.org \
  torch torchvision torchaudio faiss-gpu-cu12 --torch-backend=cu126 \
  trl peft transformers accelerate datasets wandb hydra-core 

uv pip install weave python-dotenv sentence-transformers openai

date
