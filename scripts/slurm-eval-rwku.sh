#!/bin/bash -l
#SBATCH --job-name=slurm-eval-rwku
#SBATCH --output=logs/slurm-eval-rwku-%j.log
# Request the number of gpus usint "--gres=gpu:<number>". E.g.:
#SBATCH --gres=gpu:1
# Request more time using "--time=<hours:mins:secs>". E.g.:
#SBATCH --time=01:30:00
# Request time partition "--partition=<Partition>". E.g.:
#SBATCH --partition=sc-gpu
set -euo pipefail
# Add host, time, and directory name for later troubleshooting
hostname; pwd; date
# Run the program/command
source "$HOME/anaconda3/etc/profile.d/conda.sh"
echo "Activate virtual environment (must exist)"
conda activate py312
echo "Run program in virtual environment"

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
# MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-3B-Instruct}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-microsoft/Phi-3-mini-4k-instruct}"
MODEL_LABEL="${MODEL_LABEL:-}"
SUBJECTS="${SUBJECTS:-none}"
MODEL_SLUG="$(echo "${MODEL_NAME_OR_PATH}" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]' '_' | sed 's/^_*//; s/_*$//; s/__*/_/g')"
SUBJECTS_SLUG="$(echo "${SUBJECTS}" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]' '_' | sed 's/^_*//; s/_*$//; s/__*/_/g')"
if [[ -z "${SUBJECTS_SLUG}" || "${SUBJECTS_SLUG}" == "none" || "${SUBJECTS_SLUG}" == "all" ]]; then
  SUBJECTS_SLUG="all"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/eval_rwku/${MODEL_SLUG}_${SUBJECTS_SLUG}}"
RUN_FORGET_SET="${RUN_FORGET_SET:-True}"
RUN_NEIGHBOR_SET="${RUN_NEIGHBOR_SET:-True}"
RUN_MIA_SET="${RUN_MIA_SET:-True}"
RUN_UTILITY_SET="${RUN_UTILITY_SET:-False}"
COMPUTE_MIA_LOSS="${COMPUTE_MIA_LOSS:-True}"
COMPUTE_MIA_ZLIB="${COMPUTE_MIA_ZLIB:-False}"
COMPUTE_MIA_MIN_K="${COMPUTE_MIA_MIN_K:-False}"
COMPUTE_MIA_MIN_K_PLUS_PLUS="${COMPUTE_MIA_MIN_K_PLUS_PLUS:-False}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
MAX_MIA_EXAMPLES="${MAX_MIA_EXAMPLES:-}"
MAX_UTILITY_EXAMPLES="${MAX_UTILITY_EXAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MIA_BATCH_SIZE="${MIA_BATCH_SIZE:-16}"
UTILITY_BATCH_SIZE="${UTILITY_BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-False}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"

add_bool_arg() {
  local value="$1"
  local enabled_arg="$2"
  local disabled_arg="$3"
  case "${value}" in
    True|true|1|yes|YES|y|Y)
      eval_args+=("${enabled_arg}")
      ;;
    False|false|0|no|NO|n|N)
      eval_args+=("${disabled_arg}")
      ;;
    *)
      echo "Invalid boolean value '${value}' for ${enabled_arg}/${disabled_arg}" >&2
      exit 2
      ;;
  esac
}

eval_args=(
  "${REPO_DIR}/eval/rwku/rwku.py"
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --subjects "${SUBJECTS}"
  --batch_size "${BATCH_SIZE}"
  --mia_batch_size "${MIA_BATCH_SIZE}"
  --utility_batch_size "${UTILITY_BATCH_SIZE}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --torch_dtype "${TORCH_DTYPE}"
)

add_bool_arg "${RUN_FORGET_SET}" --run_forget_set --no-run_forget_set
add_bool_arg "${RUN_NEIGHBOR_SET}" --run_neighbor_set --no-run_neighbor_set
add_bool_arg "${RUN_MIA_SET}" --run_mia_set --no-run_mia_set
add_bool_arg "${RUN_UTILITY_SET}" --run_utility_set --no-run_utility_set
add_bool_arg "${COMPUTE_MIA_LOSS}" --compute_mia_loss --no-compute_mia_loss
add_bool_arg "${COMPUTE_MIA_ZLIB}" --compute_mia_zlib --no-compute_mia_zlib
add_bool_arg "${COMPUTE_MIA_MIN_K}" --compute_mia_min_k --no-compute_mia_min_k
add_bool_arg "${COMPUTE_MIA_MIN_K_PLUS_PLUS}" --compute_mia_min_k_plus_plus --no-compute_mia_min_k_plus_plus
add_bool_arg "${TRUST_REMOTE_CODE}" --trust_remote_code --no-trust_remote_code

if [[ -n "${HF_TOKEN}" ]]; then
  eval_args+=(--hf_token "${HF_TOKEN}")
fi

if [[ -n "${MODEL_LABEL}" ]]; then
  eval_args+=(--model_label "${MODEL_LABEL}")
fi

if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  eval_args+=(--attn_implementation "${ATTN_IMPLEMENTATION}")
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
