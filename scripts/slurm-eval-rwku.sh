#!/bin/bash -l
#SBATCH --job-name=slurm-eval-rwku
#SBATCH --output=logs/slurm-eval-rwku-%j.log
# Request the number of gpus usint "--gres=gpu:<number>". E.g.:
#SBATCH --gres=gpu:2
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
cd "${REPO_DIR}"
echo "Evaluate RWKU using config/eval.yaml"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0}"
GPU_COUNT="${#GPU_IDS[@]}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Detected ${GPU_COUNT} visible GPU(s)"
CLI_OVERRIDES=("$@")

run_rwku_eval() {
  local -a overrides=("$@")

  if [[ "${GPU_COUNT}" -le 1 ]]; then
    python "${REPO_DIR}/eval/rwku/rwku.py" "${overrides[@]}"
  else
    pids=()
    for SHARD_INDEX in "${!GPU_IDS[@]}"; do
      GPU_ID="${GPU_IDS[${SHARD_INDEX}]}"
      echo "Starting eval shard ${SHARD_INDEX}/${GPU_COUNT} on GPU ${GPU_ID}"
      CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${REPO_DIR}/eval/rwku/rwku.py" \
        "${overrides[@]}" \
        evaluation.shard.index="${SHARD_INDEX}" \
        evaluation.shard.count="${GPU_COUNT}" &
      pids+=("$!")
    done

    for pid in "${pids[@]}"; do
      wait "${pid}"
    done

    python "${REPO_DIR}/eval/rwku/merge_shards.py" \
      "${overrides[@]}" \
      evaluation.shard.index=0 \
      evaluation.shard.count="${GPU_COUNT}"
  fi
}

checkpoint_state_value() {
  local model_dir="$1"
  local key="$2"
  local state_path="${model_dir}/trainer_state.json"
  if [[ ! -f "${state_path}" ]]; then
    echo "null"
    return
  fi
  python -c 'import json, sys; value = json.load(open(sys.argv[1], encoding="utf-8")).get(sys.argv[2]); print("null" if value is None else value)' "${state_path}" "${key}"
}

if [[ -n "${CHECKPOINT_ROOT:-}" ]]; then
  CHECKPOINT_ROOT="${CHECKPOINT_ROOT%/}"
  if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
    echo "CHECKPOINT_ROOT does not exist: ${CHECKPOINT_ROOT}" >&2
    exit 1
  fi

  CHECKPOINT_DIRS=()
  while IFS= read -r checkpoint_dir; do
    CHECKPOINT_DIRS+=("${checkpoint_dir}")
  done < <(find "${CHECKPOINT_ROOT}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)

  if [[ "${INCLUDE_FINAL_MODEL:-false}" == "true" && -d "${CHECKPOINT_ROOT}/final_model" ]]; then
    CHECKPOINT_DIRS+=("${CHECKPOINT_ROOT}/final_model")
  fi
  if [[ "${#CHECKPOINT_DIRS[@]}" -eq 0 ]]; then
    echo "No checkpoint-* directories found under ${CHECKPOINT_ROOT}" >&2
    exit 1
  fi

  RUN_LABEL="$(basename "${CHECKPOINT_ROOT}")"
  TRAINING_CONFIG_PATH="${CHECKPOINT_ROOT}/hydra_config.yaml"
  echo "Evaluating ${#CHECKPOINT_DIRS[@]} checkpoint(s) under ${CHECKPOINT_ROOT}"
  for MODEL_DIR in "${CHECKPOINT_DIRS[@]}"; do
    CHECKPOINT_LABEL="$(basename "${MODEL_DIR}")"
    CHECKPOINT_GLOBAL_STEP="$(checkpoint_state_value "${MODEL_DIR}" global_step)"
    CHECKPOINT_TOKENS="$(checkpoint_state_value "${MODEL_DIR}" num_input_tokens_seen)"
    EVAL_OUTPUT_DIR="${MODEL_DIR}/eval_rwku"
    echo "Evaluating checkpoint: ${MODEL_DIR}"
    echo "Checkpoint global_step=${CHECKPOINT_GLOBAL_STEP}, num_input_tokens_seen=${CHECKPOINT_TOKENS}"
    run_rwku_eval \
      "${CLI_OVERRIDES[@]}" \
      "evaluation.model_name_or_path=${MODEL_DIR}" \
      "evaluation.output_dir=${EVAL_OUTPUT_DIR}" \
      "evaluation.model.label=${CHECKPOINT_LABEL}" \
      "evaluation.training_config_path=${TRAINING_CONFIG_PATH}" \
      "evaluation.checkpoint.path=${MODEL_DIR}" \
      "evaluation.checkpoint.label=${CHECKPOINT_LABEL}" \
      "evaluation.checkpoint.global_step=${CHECKPOINT_GLOBAL_STEP}" \
      "evaluation.checkpoint.num_input_tokens_seen=${CHECKPOINT_TOKENS}" \
      "evaluation.wandb.link_to_training_run=false" \
      "evaluation.wandb.log_model_artifact=true" \
      "evaluation.wandb.run_name=rwku-${RUN_LABEL}-${CHECKPOINT_LABEL}" \
      "evaluation.wandb.artifact_name=rwku-${RUN_LABEL}-${CHECKPOINT_LABEL}"
  done
else
  run_rwku_eval "${CLI_OVERRIDES[@]}"
fi
echo "Finished RWKU evaluation"
# Add time for later troubleshooting
date
