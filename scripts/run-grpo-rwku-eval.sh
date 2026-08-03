#!/bin/bash -l
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(dirname "${SCRIPT_DIR}")}"
cd "${REPO_DIR}"

TRAIN_SCRIPT="${TRAIN_SLURM_SCRIPT:-${REPO_DIR}/scripts/run-grpo.sh}"
EVAL_SCRIPT="${EVAL_SLURM_SCRIPT:-${REPO_DIR}/scripts/eval-rwku.sh}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-unlearning}"
INCLUDE_FINAL_MODEL="${INCLUDE_FINAL_MODEL:-false}"

if [[ -n "${RUN_NAME:-}" ]]; then
  TRAIN_JOB_ID="$(sbatch --parsable --export="ALL,RUN_NAME=${RUN_NAME}" "${TRAIN_SCRIPT}" "$@")"
else
  TRAIN_JOB_ID="$(sbatch --parsable "${TRAIN_SCRIPT}" "$@")"
  RUN_NAME="${RUN_NAME_PREFIX}-${TRAIN_JOB_ID}"
fi
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-outputs/${RUN_NAME}}"

EVAL_JOB_ID="$(
  sbatch --parsable \
    --dependency="afterok:${TRAIN_JOB_ID}" \
    --export="ALL,CHECKPOINT_ROOT=${CHECKPOINT_ROOT},INCLUDE_FINAL_MODEL=${INCLUDE_FINAL_MODEL}" \
    "${EVAL_SCRIPT}"
)"

echo "Submitted training job: ${TRAIN_JOB_ID}"
echo "Training run name: ${RUN_NAME}"
echo "Submitted RWKU eval job: ${EVAL_JOB_ID}"
echo "Eval dependency: afterok:${TRAIN_JOB_ID}"
echo "Eval checkpoint root: ${CHECKPOINT_ROOT}"
