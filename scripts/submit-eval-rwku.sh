#!/bin/bash -l
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
cd "${REPO_DIR}"

EVAL_SCRIPT="${EVAL_SLURM_SCRIPT:-${REPO_DIR}/scripts/slurm-eval-rwku.sh}"
INCLUDE_FINAL_MODEL="${INCLUDE_FINAL_MODEL:-true}"

if [[ -z "${CHECKPOINT_ROOT:-}" ]]; then
  if [[ -z "${RUN_NAME:-}" ]]; then
    echo "Set RUN_NAME=unlearning-<job_id> or CHECKPOINT_ROOT=outputs/<run_name>." >&2
    exit 2
  fi
  CHECKPOINT_ROOT="outputs/${RUN_NAME}"
fi

CHECKPOINT_ROOT="${CHECKPOINT_ROOT%/}"
if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
  echo "CHECKPOINT_ROOT does not exist: ${CHECKPOINT_ROOT}" >&2
  exit 1
fi

EVAL_JOB_ID="$(
  sbatch --parsable \
    --export="ALL,CHECKPOINT_ROOT=${CHECKPOINT_ROOT},INCLUDE_FINAL_MODEL=${INCLUDE_FINAL_MODEL}" \
    "${EVAL_SCRIPT}" \
    "$@"
)"

echo "Submitted RWKU eval job: ${EVAL_JOB_ID}"
echo "Eval checkpoint root: ${CHECKPOINT_ROOT}"
echo "Include final model: ${INCLUDE_FINAL_MODEL}"
