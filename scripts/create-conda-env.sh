#!/bin/bash -l
#SBATCH --job-name=create-conda-env
#SBATCH --output=/storage/scratch/lv13/lv13594/logs/create-conda-env-%j.log
#SBATCH --qos=hopper
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -e

hostname
pwd
date

STORAGE_DIR="/storage/scratch/lv13/lv13594"
CONDA_DIR="$STORAGE_DIR/anaconda3"
TRAIN_ENV_PATH="${TRAIN_ENV_PATH:-$STORAGE_DIR/anaconda3/envs/py312_cu118}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.6.0}"
TORCH_BACKEND="${TORCH_BACKEND:-cu118}"
TRL_VERSION="${TRL_VERSION:-1.5.1}"

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

echo "Checking environment at $TRAIN_ENV_PATH"

if [ ! -f "$TRAIN_ENV_PATH/bin/python" ]; then
    echo "Creating environment"
    conda create -y -p "$TRAIN_ENV_PATH" python=3.12.3 pip
else
    echo "Environment already exists"
fi

echo "Activating environment"
conda activate "$TRAIN_ENV_PATH"

echo "Installing uv"
python -m pip install --upgrade uv

echo "Installing dependencies"

module load gcc/12.3_rhel8
module load cuda/11.8_rhel8

uv pip install \
    --reinstall \
    --torch-backend="${TORCH_BACKEND}" \
    "torch==${PYTORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"

uv pip install \
    --torch-backend="${TORCH_BACKEND}" \
    "torch==${PYTORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    "trl==${TRL_VERSION}" \
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
    rouge==1.0.1 \
    faiss-cpu \
    ninja \
    wheel \
    setuptools \
    packaging \
    psutil

echo "FlashAttention-2 build settings"

export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export MAX_JOBS=1
export NVCC_THREADS=1
export FLASH_ATTN_CUDA_ARCHS=90

echo "MAX_JOBS=$MAX_JOBS"
echo "NVCC_THREADS=$NVCC_THREADS"
echo "FLASH_ATTN_CUDA_ARCHS=$FLASH_ATTN_CUDA_ARCHS"

# Keep FlashAttention disabled unless the cluster CUDA module and PyTorch
# version are known to match.
# uv pip install flash-attn==2.8.3.post1 --no-build-isolation

echo "Installation completed"
python -c "import torch, trl; import torch.distributed.fsdp as fsdp; print(torch.__version__, torch.version.cuda); print(trl.__version__); print('FSDPModule', hasattr(fsdp, 'FSDPModule')); torch.cuda.set_device(0); print(torch.ones(1, device='cuda'))"
date
