#!/bin/bash -l
#SBATCH --job-name=evals
#SBATCH --output=logs/evals-%j.log
#SBATCH --time=00:30:00
#SBATCH --partition=hopper
#SBATCH --qos=hopper

# Example:
#   scripts/submit-eval-rwku.sh unlearning-12345
#   scripts/submit-eval-rwku.sh outputs/unlearning-12345 outputs/unlearning-67890 -- evaluation.batch_size=8
#   scripts/submit-eval-rwku.sh outputs/unlearning-12345/checkpoint-150

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
  echo "Usage: scripts/submit-eval-rwku.sh RUN_OR_CHECKPOINT_PATH [...] [-- EVAL_OVERRIDE ...]" >&2
  exit 1
fi

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

resolve_run_path() {
  local spec="$1"
  local path

  if [[ -d "${spec}" ]]; then
    path="${spec}"
  elif [[ -d "outputs/${spec}" ]]; then
    path="outputs/${spec}"
  else
    echo "Could not find run '${spec}' as a directory or outputs/${spec}" >&2
    return 1
  fi

  printf '%s\n' "${path%/}"
}

for run_spec in "${RUN_SPECS[@]}"; do
  run_path="$(resolve_run_path "${run_spec}")"

  if [[ -f "${run_path}/hydra_config.yaml" ]]; then
    run_name="$(basename "${run_path}")"
    eval_job_id="$(
      sbatch --parsable \
        --export="ALL,CHECKPOINT_ROOT=${run_path},INCLUDE_FINAL_MODEL=${INCLUDE_FINAL_MODEL}" \
        "${EVAL_SCRIPT}" \
        "${EVAL_OVERRIDES[@]}"
    )"
    echo "Submitted RWKU eval job: ${eval_job_id} (${run_name})"
    continue
  fi

  run_root="$(dirname "${run_path}")"
  training_config="${run_root}/hydra_config.yaml"
  if [[ ! -f "${training_config}" ]]; then
    echo "Path is neither a run root nor a checkpoint/final_model under a run: ${run_path}" >&2
    exit 1
  fi

  checkpoint_label="$(basename "${run_path}")"
  subjects="$(get_experiment_value "forget_concept" "${training_config}")"
  eval_output_dir="${run_path}/eval_rwku"
  eval_job_id="$(
    sbatch --parsable \
      "${EVAL_SCRIPT}" \
      "evaluation.model_name_or_path=${run_path}" \
      "evaluation.output_dir=${eval_output_dir}" \
      "evaluation.subjects=${subjects}" \
      "evaluation.model.label=${checkpoint_label}" \
      "evaluation.training_config_path=${training_config}" \
      "evaluation.checkpoint.path=${run_path}" \
      "evaluation.checkpoint.label=${checkpoint_label}" \
      "evaluation.wandb.run_name=rwku-$(basename "${run_root}")-${checkpoint_label}" \
      "evaluation.wandb.artifact_name=rwku-$(basename "${run_root}")-${checkpoint_label}" \
      "${EVAL_OVERRIDES[@]}"
  )"
  echo "Submitted RWKU eval job: ${eval_job_id} ($(basename "${run_root}")/${checkpoint_label})"
done
