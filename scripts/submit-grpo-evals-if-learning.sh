#!/bin/bash -l
#SBATCH --job-name=eval-gate
#SBATCH --output=logs/eval-gate-%j.log
#SBATCH --time=00:05:00
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
CHECKPOINT_ROOT="${1:?Usage: submit-grpo-evals-if-learning.sh CHECKPOINT_ROOT CONCEPT}"
CONCEPT="${2:?Usage: submit-grpo-evals-if-learning.sh CHECKPOINT_ROOT CONCEPT}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT%/}"
LOW_REWARD_STOP_MARKER="${LOW_REWARD_STOP_MARKER:-low_reward_stop.json}"
MARKER_PATH="${CHECKPOINT_ROOT}/${LOW_REWARD_STOP_MARKER}"

cd "${REPO_DIR}"

holdout_output_dir() {
  local concept="$1"
  local model_name_or_path="$2"
  local concept_part
  local model_part

  concept_part="$(echo "${concept// /-}" | tr '[:upper:]' '[:lower:]')"
  model_part="${model_name_or_path//\//-}"
  echo "outputs/hold_out_styles/${concept_part}-${model_part}"
}

if [[ -f "${MARKER_PATH}" ]]; then
  echo "Skipping RWKU and hold-out evals because low-reward stop marker exists: ${MARKER_PATH}"
  cat "${MARKER_PATH}"
  exit 0
fi

RWKU_OUTPUT_DIR="${CHECKPOINT_ROOT}/final_model/eval_rwku"
HOLDOUT_OUTPUT_DIR="$(holdout_output_dir "${CONCEPT}" "${CHECKPOINT_ROOT}/final_model")"

if [[ -f "${RWKU_OUTPUT_DIR}/rwku_summary_table.csv" ]]; then
  rwku_job_id="done"
else
  rwku_job_id="$(
    sbatch --parsable \
      --export="ALL,CHECKPOINT_ROOT=${CHECKPOINT_ROOT},ONLY_FINAL_MODEL=true" \
      scripts/eval-rwku.sh
  )"
fi

if [[ -f "${HOLDOUT_OUTPUT_DIR}/metrics.csv" && -f "${HOLDOUT_OUTPUT_DIR}/summary.csv" ]]; then
  holdout_job_id="done"
else
  holdout_job_id="$(
    sbatch --parsable \
      --job-name="holdout" \
      --output="logs/holdout-%j.log" \
      --gres=gpu:1 \
      --time="01:30:00" \
      --partition="sc-gpu" \
      --wrap="cd ${REPO_DIR} && source \$HOME/anaconda3/etc/profile.d/conda.sh && conda activate py312 && python eval/hold_out_styles/generate-and-analyze-completions.py concept='${CONCEPT}' model_name_or_path='${CHECKPOINT_ROOT}/final_model'"
  )"
fi

echo "Submitted RWKU eval: ${rwku_job_id}"
echo "Submitted hold-out eval: ${holdout_job_id}"
