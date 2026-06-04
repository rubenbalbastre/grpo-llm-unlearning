set -euo pipefail

cd /workspace/machine-unlearning-llm

export PIP_CACHE_DIR=/workspace/.cache/pip
export UV_CACHE_DIR=/workspace/.cache/uv
export TMPDIR=/workspace/.tmp

mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$TMPDIR"

# Remove the too-new HF stack
python -m pip uninstall -y transformers trl peft tokenizers accelerate

# Reinstall a coherent pre-Transformers-5 stack.
# This should avoid the finegrained_fp8 import path causing your crash.
python -m pip install --no-cache-dir --root-user-action=ignore \
  "transformers==4.56.2" \
  "tokenizers" \
  "trl" \
  "peft==0.17.1" \
  "accelerate" \
  "datasets" \
  "wandb" \
  "hydra-core" \
  "weave" \
  "python-dotenv" \
  "sentence-transformers" \
  "openai" \
  "rouge-score" \
  "faiss-cpu"