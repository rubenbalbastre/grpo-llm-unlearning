#!/bin/bash -l
#SBATCH --job-name=eval-rwku
#SBATCH --output=logs/eval-rwku-%j.log
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --partition=hopper
#SBATCH --qos=hopper
set -euo pipefail

ENV_DIR="${ENV_DIR:-/storage/scratch/lv13/lv13594/}"
REPO_DIR="${REPO_DIR:-/storage/scratch/lv13/lv13594/fresh-repo/grpo-llm-unlearning}"
source "$ENV_DIR/anaconda3_bis/etc/profile.d/conda.sh"
conda activate py312_cu118_bis
cd "${REPO_DIR}"

if [[ -n "${SKIP_IF_MARKER:-}" && -f "${SKIP_IF_MARKER}" ]]; then
  echo "Skipping RWKU evaluation because marker exists: ${SKIP_IF_MARKER}"
  exit 0
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0}"
GPU_COUNT="${#GPU_IDS[@]}"

run_rwku() {
  local -a overrides=("$@")

  if [[ "${GPU_COUNT}" -eq 1 ]]; then
    python eval/rwku/rwku.py "${overrides[@]}"
    return
  fi

  pids=()
  for shard_index in "${!GPU_IDS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_IDS[${shard_index}]}" \
      python eval/rwku/rwku.py \
        "${overrides[@]}" \
        evaluation.shard.index="${shard_index}" \
        evaluation.shard.count="${GPU_COUNT}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
  python eval/rwku/merge_shards.py \
    "${overrides[@]}" \
    evaluation.shard.index=0 \
    evaluation.shard.count="${GPU_COUNT}"
}

if [[ -z "${CHECKPOINT_ROOT:-}" ]]; then
  run_rwku "$@"
  exit 0
fi

CHECKPOINT_ROOT="${CHECKPOINT_ROOT%/}"
MODEL_DIR="${CHECKPOINT_ROOT}/final_model"
TRAINING_CONFIG_PATH="${CHECKPOINT_ROOT}/hydra_config.yaml"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Final model directory not found: ${MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -f "${TRAINING_CONFIG_PATH}" ]]; then
  echo "Training config not found: ${TRAINING_CONFIG_PATH}" >&2
  exit 1
fi

SUBJECT="$(python -c 'import sys; from omegaconf import OmegaConf; cfg = OmegaConf.load(sys.argv[1]); print(cfg.experiment.forget_concept)' "${TRAINING_CONFIG_PATH}")"
RUN_LABEL="$(basename "${CHECKPOINT_ROOT}")"

run_rwku \
  "$@" \
  "evaluation.model_name_or_path=${MODEL_DIR}" \
  "evaluation.output_dir=${MODEL_DIR}/eval_rwku" \
  "evaluation.subjects=${SUBJECT}" \
  "evaluation.model.label=final_model" \
  "evaluation.training_config_path=${TRAINING_CONFIG_PATH}" \
  "evaluation.wandb.log_artifact=true" \
  "evaluation.wandb.log_model_artifact=false" \
  "evaluation.wandb.run_name=rwku-${RUN_LABEL}-final_model" \
  "evaluation.wandb.artifact_name=rwku-${RUN_LABEL}-final_model"
