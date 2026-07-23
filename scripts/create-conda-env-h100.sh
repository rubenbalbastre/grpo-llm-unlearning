#!/bin/bash -l
#SBATCH --job-name=create-conda-env
#SBATCH --output=logs/create-conda-env-%j.log
#SBATCH --qos=hopper
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
hostname; pwd; date

echo "Starting"

# Fixes the 'RECORD file is invalid' (os error 5) network filesystem errors
export UV_CONCURRENT_WRITES=1
export UV_NO_CACHE=1
rm -rf ~/.cache/uv

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

echo "Installing high-speed dependency manager and compilation tools"
pip install --upgrade uv
# ninja and wheel are strictly mandatory for flash-attn --no-build-isolation
uv pip install ninja wheel 

echo "Executing proper multi-repository installation"

# FIXED: --torch-backend placed correctly at the top
uv pip install --torch-backend=cu121 \
  torch torchvision torchaudio faiss-gpu-cu12 \
  trl peft transformers accelerate deepspeed datasets wandb hydra-core \
  vllm

echo "Building local hardware optimizations"
# The cluster drivers will now link cleanly via your active cuda/12.1 module
uv pip install flash-attn --no-build-isolation

echo "Installing remaining utilities"
uv pip install weave python-dotenv sentence-transformers openai rouge-score
date