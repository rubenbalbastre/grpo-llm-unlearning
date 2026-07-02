#!/bin/bash -l
#SBATCH --job-name=evals
#SBATCH --output=logs/evals-%j.log
#SBATCH --time=00:30:00
#SBATCH --partition=sc-gpu
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
cd "${REPO_DIR}"

EVAL_SCRIPT="${EVAL_SLURM_SCRIPT:-${REPO_DIR}/scripts/slurm-eval-rwku.sh}"
INCLUDE_FINAL_MODEL="${INCLUDE_FINAL_MODEL:-true}"

get_experiment_value() {
  local key="$1"
  local config_path="$2"

  awk -v key="${key}" '
    /^experiment:/ { in_experiment = 1; next }
    /^[^[:space:]]/ { in_experiment = 0 }
    in_experiment && $1 == key ":" {
      sub("^[[:space:]]*" key ":[[:space:]]*", "")
      print
      exit
    }
  ' "${config_path}"
}

matched_runs=0

while IFS= read -r hydra_config; do
  notes="$(get_experiment_value "notes" "${hydra_config}")"
  if [[ "${notes}" != "apply-chat-template" ]]; then
    continue
  fi

  CHECKPOINT_ROOT="$(dirname "${hydra_config}")"
  RUN_NAME="$(basename "${CHECKPOINT_ROOT}")"
  CONCEPT="$(get_experiment_value "forget_concept" "${hydra_config}")"

  EVAL_JOB_ID="$(
    sbatch --parsable \
      --export="ALL,CHECKPOINT_ROOT=${CHECKPOINT_ROOT},INCLUDE_FINAL_MODEL=${INCLUDE_FINAL_MODEL}" \
      "${EVAL_SCRIPT}" \
      "$@"
  )"

  echo "Submitted RWKU eval job: ${EVAL_JOB_ID} (${RUN_NAME}, concept=${CONCEPT})"
  matched_runs=$((matched_runs + 1))
done < <(find outputs -mindepth 2 -maxdepth 2 -type f -name 'hydra_config.yaml' | sort -V)

if [[ "${matched_runs}" -eq 0 ]]; then
  echo "No runs found with hydra.experiment.notes=apply-chat-template" >&2
  exit 1
fi
