#!/bin/bash -l
#SBATCH --job-name=create-conda-env
#SBATCH --output=/storage/scratch/lv13/lv13594/logs/create-conda-env-%j.log
#SBATCH --qos=hopper
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -e

hostname
pwd
date

STORAGE_DIR="/storage/scratch/lv13/lv13594"
CONDA_DIR="$STORAGE_DIR/anaconda3"
ENV_PATH="$STORAGE_DIR/anaconda3/envs/py312"

mkdir -p "$STORAGE_DIR/logs"
mkdir -p "$STORAGE_DIR/tmp"

export TMPDIR="$STORAGE_DIR/tmp"
export UV_NO_CACHE=1
export UV_LINK_MODE=copy

echo "Installing Anaconda"

if [ ! -f "$CONDA_DIR/bin/conda" ]; then
    wget -O "$STORAGE_DIR/anaconda-installer.sh" \
        https://repo.anaconda.com/archive/Anaconda3-2025.06-1-Linux-x86_64.sh

    bash "$STORAGE_DIR/anaconda-installer.sh" \
        -b \
        -p "$CONDA_DIR"
else
    echo "Anaconda already installed"
fi

source "$CONDA_DIR/etc/profile.d/conda.sh"

echo "Checking environment at $ENV_PATH"

if [ ! -f "$ENV_PATH/bin/python" ]; then
    echo "Creating environment"
    conda create -y -p "$ENV_PATH" python=3.12.3 pip
else
    echo "Environment already exists"
fi

echo "Activating environment"
conda activate "$ENV_PATH"

echo "Installing uv"
python -m pip install --upgrade uv

echo "Installing dependencies"

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

echo "Installing FlashAttention-2"

module load gcc/12.3_rhel8

export CC="$(command -v gcc)"
export CXX="$(command -v g++)"

gcc --version
g++ --version

export MAX_JOBS=4
uv pip install flash-attn --no-build-isolation

echo "Installation completed"
python -c "import torch; print(torch.__version__, torch.version.cuda)"
date