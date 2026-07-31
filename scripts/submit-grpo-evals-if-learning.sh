#!/bin/bash -l
#SBATCH --job-name=eval-gate
#SBATCH --output=logs/eval-gate-%j.log
#SBATCH --time=00:05:00
set -euo pipefail

STORAGE_DIR="${STORAGE_DIR:-/storage/scratch/lv13/lv13594}"
export HF_HOME="${HF_HOME:-${STORAGE_DIR}/huggingface}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HOME}/.cache/huggingface/token}"
CONDA_DIR="${CONDA_DIR:-${STORAGE_DIR}/anaconda3}"
TRAIN_ENV_PATH="${TRAIN_ENV_PATH:-${CONDA_DIR}/envs/py312_cu118}"
REPO_DIR="${REPO_DIR:-${STORAGE_DIR}/machine-unlearning-llm}"
CHECKPOINT_ROOT="${1:?Usage: submit-grpo-evals-if-learning.sh CHECKPOINT_ROOT CONCEPT}"
CONCEPT="${2:?Usage: submit-grpo-evals-if-learning.sh CHECKPOINT_ROOT CONCEPT}"
STORAGE_ROOT="${3:-${STORAGE_ROOT:-.}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT%/}"
LOW_REWARD_STOP_MARKER="${LOW_REWARD_STOP_MARKER:-low_reward_stop.json}"
MARKER_PATH="${CHECKPOINT_ROOT}/${LOW_REWARD_STOP_MARKER}"
RUN_RWKU_EVAL="${RUN_RWKU_EVAL:-true}"
RUN_HOLDOUT_EVAL="${RUN_HOLDOUT_EVAL:-true}"

cd "${REPO_DIR}"

flag_enabled() {
  [[ "$1" == "true" ]]
}

if ! flag_enabled "${RUN_RWKU_EVAL}" && ! flag_enabled "${RUN_HOLDOUT_EVAL}"; then
  echo "Skipping eval gate because both RUN_RWKU_EVAL and RUN_HOLDOUT_EVAL are disabled."
  exit 0
fi

if [[ -f "${MARKER_PATH}" ]]; then
  echo "Skipping enabled evals because low-reward stop marker exists: ${MARKER_PATH}"
  cat "${MARKER_PATH}"
  exit 0
fi

RWKU_OUTPUT_DIR="${CHECKPOINT_ROOT}/final_model/eval_rwku"
HOLDOUT_OUTPUT_DIR="${CHECKPOINT_ROOT}/final_model/hold_out_eval"

if ! flag_enabled "${RUN_RWKU_EVAL}"; then
  rwku_job_id="disabled"
elif [[ -f "${RWKU_OUTPUT_DIR}/rwku_summary_table.csv" ]]; then
  rwku_job_id="done"
else
  rwku_job_id="$(
    sbatch --parsable \
      --export="ALL,CHECKPOINT_ROOT=${CHECKPOINT_ROOT},ONLY_FINAL_MODEL=true" \
      scripts/eval-rwku.sh
  )"
fi

if ! flag_enabled "${RUN_HOLDOUT_EVAL}"; then
  holdout_job_id="disabled"
elif [[ -f "${HOLDOUT_OUTPUT_DIR}/metrics.csv" && -f "${HOLDOUT_OUTPUT_DIR}/summary.csv" ]]; then
  holdout_job_id="done"
else
  holdout_job_id="$(
    sbatch --parsable \
      --job-name="holdout" \
      --output="logs/holdout-%j.log" \
      --gres=gpu:1 \
      --time="01:30:00" \
      --partition="hopper" \
      --qos="hopper" \
      --wrap="cd ${REPO_DIR} && source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate ${TRAIN_ENV_PATH} && python eval/hold_out_styles/generate-and-analyze-completions.py concept='${CONCEPT}' model_name_or_path='${CHECKPOINT_ROOT}/final_model' output_dir='${HOLDOUT_OUTPUT_DIR}' paths.storage_root='${STORAGE_ROOT}'"
  )"
fi

echo "Submitted RWKU eval: ${rwku_job_id}"
echo "Submitted hold-out eval: ${holdout_job_id}"
