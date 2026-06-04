set -euo pipefail

cd /workspace/machine-unlearning-llm

export PIP_CACHE_DIR=/workspace/.cache/pip
export UV_CACHE_DIR=/workspace/.cache/uv
export TMPDIR=/workspace/.tmp

mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$TMPDIR"

python -m pip install --no-cache-dir --root-user-action=ignore \
  "transformers" \
  "tokenizers" \
  "trl" \
  "peft" \
  "accelerate" \
  "datasets" \
  "wandb" \
  "hydra-core" \
  "weave" \
  "python-dotenv" \
  "sentence-transformers" \
  "openai" \
  "rouge-score" \
  "faiss-cpu" \
  "hf_transfer" \
  "vllm"

# python -m pip install --no-cache-dir --root-user-action=ignore flash-attn --no-build-isolation