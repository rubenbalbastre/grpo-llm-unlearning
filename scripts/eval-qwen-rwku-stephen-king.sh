#!/bin/bash -l
#SBATCH --job-name=slurm-eval-qwen-rwku
#SBATCH --output=logs/slurm-eval-qwen-rwku-%j.log
# Request the number of gpus usint "--gres=gpu:<number>". E.g.:
#SBATCH --gres=gpu:1
# Request more time using "--time=<hours:mins:secs>". E.g.:
#SBATCH --time=00:30:00
# Request time partition "--partition=<Partition>". E.g.:
#SBATCH --partition=sc-gpu
# Add host, time, and directory name for later troubleshooting
hostname; pwd; date
# Run the program/command
source "$HOME/anaconda3/etc/profile.d/conda.sh"
echo "Activate virtual environment (must exist)"
conda activate py312
echo "Run program in virtual environment"

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_LABEL="${MODEL_LABEL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/eval_rwku/qwen_stephen_king}"
SUBJECTS="${SUBJECTS:-Stephen King}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
MAX_MIA_EXAMPLES="${MAX_MIA_EXAMPLES:-}"
MAX_UTILITY_EXAMPLES="${MAX_UTILITY_EXAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"

eval_args=(
  "${REPO_DIR}/eval_rwku.py"
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --subjects "${SUBJECTS}"
  --batch_size "${BATCH_SIZE}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --torch_dtype "${TORCH_DTYPE}"
)

if [[ -n "${HF_TOKEN}" ]]; then
  eval_args+=(--hf_token "${HF_TOKEN}")
fi

if [[ -n "${MODEL_LABEL}" ]]; then
  eval_args+=(--model_label "${MODEL_LABEL}")
fi

if [[ -n "${MAX_EXAMPLES}" ]]; then
  eval_args+=(--max_examples "${MAX_EXAMPLES}")
fi

if [[ -n "${MAX_MIA_EXAMPLES}" ]]; then
  eval_args+=(--max_mia_examples "${MAX_MIA_EXAMPLES}")
fi

if [[ -n "${MAX_UTILITY_EXAMPLES}" ]]; then
  eval_args+=(--max_utility_examples "${MAX_UTILITY_EXAMPLES}")
fi

echo "Evaluating ${MODEL_NAME_OR_PATH} on RWKU subject: ${SUBJECTS}"
python "${eval_args[@]}"

echo "Wrote results to ${OUTPUT_DIR}"
# Add time for later troubleshooting
date
