#!/usr/bin/env bash
set -euo pipefail

MODEL_URL="https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
MODEL_PATH="${1:-${FASTTEXT_LID_PATH:-$HOME/models/fasttext/lid.176.ftz}}"

mkdir -p "$(dirname "$MODEL_PATH")"

if [[ -f "$MODEL_PATH" ]]; then
  echo "fastText language ID model already exists: $MODEL_PATH"
  exit 0
fi

echo "Downloading fastText language ID model to: $MODEL_PATH"
if command -v curl >/dev/null 2>&1; then
  curl -L "$MODEL_URL" -o "$MODEL_PATH"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$MODEL_PATH" "$MODEL_URL"
else
  echo "Neither curl nor wget is available." >&2
  exit 1
fi

echo "Done. Add this to your environment:"
echo "export FASTTEXT_LID_PATH=\"$MODEL_PATH\""
