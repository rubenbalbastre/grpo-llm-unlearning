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

echo "Executing proper multi-repository installation"
uv pip install \
    --torch-backend=auto \
    "trl[vllm]" \
    transformers \
    peft \
    accelerate \
    deepspeed \
    datasets \
    wandb \
    hydra-core \
    weave \
    python-dotenv \
    sentence-transformers \
    openai \
    rouge-score \
    faiss-gpu-cu12 \
    ninja \
    wheel \
    setuptools \
    packaging \
    psutil

echo "Building FlashAttention-2"
uv pip install flash-attn --no-build-isolation

date