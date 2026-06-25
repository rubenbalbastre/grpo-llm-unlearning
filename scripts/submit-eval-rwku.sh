#!/bin/bash -l
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
cd "${REPO_DIR}"

EVAL_SCRIPT="${EVAL_SLURM_SCRIPT:-${REPO_DIR}/scripts/slurm-eval-rwku.sh}"
INCLUDE_FINAL_MODEL="${INCLUDE_FINAL_MODEL:-true}"


run_names=(
  # "unlearning-4162"
  # "unlearning-4106"
  # "unlearning-4112"
  # "unlearning-4114"
  # "unlearning-4164"
  # "unlearning-4058"
  # "unlearning-4161"
  # "unlearning-4099"
  # "unlearning-4111"
  # "unlearning-4113"
  # "unlearning-4085"  
  # "unlearning-4163"
  # "unlearning-4083"
  # "unlearning-4160"
  # "unlearning-4110"
  # "unlearning-4208"
  # "unlearning-4207"
  # "unlearning-4206"
  # "unlearning-4205"
  # "unlearning-4204"
  # "unlearning-4203"
  # "unlearning-4202"
  # "unlearning-4201"
  # "unlearning-4200"
  # "unlearning-4199"
  # "unlearning-4198"
  # "unlearning-4197"
  # "unlearning-4196"
  # "unlearning-4194"
  # "unlearning-4260"
  # "unlearning-4261"
  # "unlearning-4307"
  # "unlearning-4308"
  # "unlearning-4309"
  # "unlearning-4310"
  # "unlearning-4311" run 
  # "unlearning-4312" run
  # "unlearning-4095"
  # "unlearning-4101"
  # "unlearning-4087"
  # "unlearning-4089"
  # "unlearning-4097"
  # "unlearning-4103"
  # "unlearning-4098"
  # "unlearning-4090"
  # "unlearning-4104"
  # "unlearning-4096"
  # "unlearning-4102"
  # "unlearning-4088"
  # "unlearning-4107"
  # "unlearning-4109"
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