#!/bin/bash -l
#SBATCH --job-name=create-conda-env
#SBATCH --output=logs/create-conda-env-%j.log
#SBATCH --qos=hopper
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
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

# 2. Run the unified installation explicitly locking onto pre-built cu126 binaries
uv pip install \
  torch torchvision torchaudio
  trl peft transformers accelerate deepspeed datasets wandb hydra-core \
  vllm

uv pip install weave python-dotenv sentence-transformers openai rouge-score
date
