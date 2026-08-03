#!/bin/bash -l
#SBATCH --job-name=evals
#SBATCH --output=logs/evals-%j.log
#SBATCH --time=00:30:00
#SBATCH --partition=hopper
#SBATCH --qos=hopper

# Example:
#   scripts/submit-eval-rwku.sh unlearning-12345
#   scripts/submit-eval-rwku.sh outputs/unlearning-12345 outputs/unlearning-67890 -- evaluation.batch_size=8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(dirname "${SCRIPT_DIR}")}"
EVAL_SCRIPT="${EVAL_SLURM_SCRIPT:-${REPO_DIR}/scripts/eval-rwku.sh}"
INCLUDE_FINAL_MODEL="${INCLUDE_FINAL_MODEL:-true}"

cd "${REPO_DIR}"

RUN_SPECS=()
EVAL_OVERRIDES=()

while [[ "$#" -gt 0 ]]; do
  if [[ "$1" == "--" ]]; then
    shift
    EVAL_OVERRIDES=("$@")
    break
  fi
  RUN_SPECS+=("$1")
  shift
done

if [[ "${#RUN_SPECS[@]}" -eq 0 ]]; then
  echo "Usage: scripts/submit-eval-rwku.sh RUN_OR_PATH [RUN_OR_PATH ...] [-- EVAL_OVERRIDE ...]" >&2
  exit 1
fi

resolve_checkpoint_root() {
  local spec="$1"
  local checkpoint_root

  if [[ -d "${spec}" ]]; then
    checkpoint_root="${spec}"
  elif [[ -d "outputs/${spec}" ]]; then
    checkpoint_root="outputs/${spec}"
  else
    echo "Could not find run '${spec}' as a directory or outputs/${spec}" >&2
    return 1
  fi

  checkpoint_root="${checkpoint_root%/}"
  if [[ ! -f "${checkpoint_root}/hydra_config.yaml" ]]; then
    echo "Run is missing hydra_config.yaml: ${checkpoint_root}" >&2
    return 1
  fi

  printf '%s\n' "${checkpoint_root}"
}

for run_spec in "${RUN_SPECS[@]}"; do
  checkpoint_root="$(resolve_checkpoint_root "${run_spec}")"
  run_name="$(basename "${checkpoint_root}")"
  eval_job_id="$(
    sbatch --parsable \
      --export="ALL,CHECKPOINT_ROOT=${checkpoint_root},INCLUDE_FINAL_MODEL=${INCLUDE_FINAL_MODEL}" \
      "${EVAL_SCRIPT}" \
      "${EVAL_OVERRIDES[@]}"
  )"
  echo "Submitted RWKU eval job: ${eval_job_id} (${run_name})"
done
