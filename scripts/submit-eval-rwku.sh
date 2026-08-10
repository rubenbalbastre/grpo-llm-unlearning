#!/bin/bash -l
#SBATCH --job-name=evals
#SBATCH --output=logs/evals-%j.log
#SBATCH --time=00:30:00
#SBATCH --partition=sc-gpu
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
cd "${REPO_DIR}"

EVAL_SCRIPT="${EVAL_SLURM_SCRIPT:-${REPO_DIR}/scripts/eval-rwku.sh}"
INCLUDE_FINAL_MODEL="${INCLUDE_FINAL_MODEL:-true}"
MODEL_LIST="${MODEL_LIST:-}"
MODEL_LIST_FILE="${MODEL_LIST_FILE:-}"

usage() {
  cat <<'EOF'
Usage:
  scripts/submit-eval-rwku.sh [MODEL_OR_RUN ...] [-- EVAL_OVERRIDE ...]
  MODEL_LIST="run-a,run-b" scripts/submit-eval-rwku.sh [-- EVAL_OVERRIDE ...]
  scripts/submit-eval-rwku.sh --model-list runs.txt [-- EVAL_OVERRIDE ...]

MODEL_OR_RUN can be a path under outputs/ or a basename resolved as outputs/<name>.
When no models/runs are provided, the script keeps the old behavior: auto-discover
outputs/*/hydra_config.yaml runs with experiment.notes=apply-chat-template.
EOF
}

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

add_model_spec() {
  local spec="$1"
  [[ -n "${spec}" ]] || return 0
  MODEL_SPECS+=("${spec}")
}

add_model_list_value() {
  local list_value="$1"
  local normalized="${list_value//,/ }"
  local spec
  for spec in ${normalized}; do
    add_model_spec "${spec}"
  done
}

add_model_list_file() {
  local list_path="$1"
  local line

  if [[ ! -f "${list_path}" ]]; then
    echo "Model list file does not exist: ${list_path}" >&2
    exit 1
  fi

  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    add_model_list_value "${line}"
  done < "${list_path}"
}

resolve_checkpoint_root() {
  local spec="$1"
  local checkpoint_root

  if [[ -d "${spec}" ]]; then
    checkpoint_root="${spec}"
  elif [[ -d "outputs/${spec}" ]]; then
    checkpoint_root="outputs/${spec}"
  else
    echo "Could not find model/run '${spec}' as a directory or outputs/${spec}" >&2
    return 1
  fi

  checkpoint_root="${checkpoint_root%/}"
  if [[ ! -f "${checkpoint_root}/hydra_config.yaml" ]]; then
    echo "Run is missing hydra_config.yaml and cannot be evaluated via CHECKPOINT_ROOT: ${checkpoint_root}" >&2
    return 1
  fi

  printf '%s\n' "${checkpoint_root}"
}

submit_eval_job() {
  local checkpoint_root="$1"
  local run_name
  local concept
  local eval_job_id

  run_name="$(basename "${checkpoint_root}")"
  concept="$(get_experiment_value "forget_concept" "${checkpoint_root}/hydra_config.yaml")"

  eval_job_id="$(
    sbatch --parsable \
      --export="ALL,CHECKPOINT_ROOT=${checkpoint_root},INCLUDE_FINAL_MODEL=${INCLUDE_FINAL_MODEL}" \
      "${EVAL_SCRIPT}" \
      "${EVAL_OVERRIDES[@]}"
  )"

  echo "Submitted RWKU eval job: ${eval_job_id} (${run_name}, concept=${concept})"
}

MODEL_SPECS=()
EVAL_OVERRIDES=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EVAL_OVERRIDES=("$@")
      break
      ;;
    --model|--run|--checkpoint-root)
      if [[ "$#" -lt 2 ]]; then
        echo "$1 requires a value" >&2
        exit 1
      fi
      add_model_spec "$2"
      shift 2
      ;;
    --models|--runs|--checkpoint-roots)
      shift
      while [[ "$#" -gt 0 && "$1" != "--" ]]; do
        add_model_spec "$1"
        shift
      done
      ;;
    --model-list|--run-list|--checkpoint-root-list)
      if [[ "$#" -lt 2 ]]; then
        echo "$1 requires a file path" >&2
        exit 1
      fi
      MODEL_LIST_FILE="$2"
      shift 2
      ;;
    *=*)
      EVAL_OVERRIDES+=("$1")
      shift
      ;;
    *)
      add_model_spec "$1"
      shift
      ;;
  esac
done

if [[ -n "${MODEL_LIST}" ]]; then
  add_model_list_value "${MODEL_LIST}"
fi

if [[ -n "${MODEL_LIST_FILE}" ]]; then
  add_model_list_file "${MODEL_LIST_FILE}"
fi

matched_runs=0

if [[ "${#MODEL_SPECS[@]}" -gt 0 ]]; then
  for model_spec in "${MODEL_SPECS[@]}"; do
    CHECKPOINT_ROOT="$(resolve_checkpoint_root "${model_spec}")"
    submit_eval_job "${CHECKPOINT_ROOT}"
    matched_runs=$((matched_runs + 1))
  done
  exit 0
fi

while IFS= read -r hydra_config; do
  notes="$(get_experiment_value "notes" "${hydra_config}")"
  if [[ "${notes}" != "apply-chat-template" ]]; then
    continue
  fi

  CHECKPOINT_ROOT="$(dirname "${hydra_config}")"
  submit_eval_job "${CHECKPOINT_ROOT}"
  matched_runs=$((matched_runs + 1))
done < <(find outputs -mindepth 2 -maxdepth 2 -type f -name 'hydra_config.yaml' | sort -V)

if [[ "${matched_runs}" -eq 0 ]]; then
  echo "No runs found with hydra.experiment.notes=apply-chat-template" >&2
  exit 1
fi
