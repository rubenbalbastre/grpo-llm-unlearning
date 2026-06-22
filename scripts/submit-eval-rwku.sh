#!/bin/bash -l
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
cd "${REPO_DIR}"

EVAL_SCRIPT="${EVAL_SLURM_SCRIPT:-${REPO_DIR}/scripts/slurm-eval-rwku.sh}"
INCLUDE_FINAL_MODEL="${INCLUDE_FINAL_MODEL:-true}"


run_names=(
  "unlearning-4162"
  "unlearning-4106"
  "unlearning-4112"
  "unlearning-4114"
  "unlearning-4164"
  "unlearning-4058"
  "unlearning-4161"
  "unlearning-4099"
  "unlearning-4111"
  "unlearning-4113"
  "unlearning-4085"  
  "unlearning-4163"
  "unlearning-4083"
  "unlearning-4160"
  "unlearning-4110"
)

for RUN_NAME in "${run_names[@]}"; do

  CHECKPOINT_ROOT="outputs/${RUN_NAME}"

  EVAL_JOB_ID="$(
    sbatch --parsable \
      --export="ALL,CHECKPOINT_ROOT=${CHECKPOINT_ROOT},INCLUDE_FINAL_MODEL=${INCLUDE_FINAL_MODEL}" \
      "${EVAL_SCRIPT}" \
      "$@"
  )"

  echo "Submitted RWKU eval job: ${EVAL_JOB_ID}"

done