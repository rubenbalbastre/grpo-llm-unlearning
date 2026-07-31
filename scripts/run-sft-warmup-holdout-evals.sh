#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
cd "${REPO_DIR}"

LOW_REWARD_STOP_MARKER="${LOW_REWARD_STOP_MARKER:-low_reward_stop.json}"
STORAGE_ROOT="${STORAGE_ROOT:-.}"
DATA_ROOT="${STORAGE_ROOT}/data"
OUTPUT_ROOT="${STORAGE_ROOT}/outputs"

original_models=(
  # "Qwen/Qwen2.5-0.5B-Instruct"
  # "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
)

targets=(
  "Jennifer Lopez"
  "Tony Blair"
  "Marlon Brando"
  "Bruce Lee"
  "Serena Williams"
  "John D. Rockefeller"
  "Tom Clancy"
  "Vincent van Gogh"
  "Karl Marx"
  "Confucius"
)

underscore_slug() {
  echo "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
}

dependency_arg() {
  local dependency="$1"

  if [[ -n "${dependency}" ]]; then
    echo "--dependency=afterok:${dependency}"
  fi
}

submit_warmup_holdout() {
  local model="$1"
  local target="$2"
  local serial_dependency="$3"
  local run_name="r2warmup_$(underscore_slug "${model}")_$(underscore_slug "${target}")"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  local final_model="${run_dir}/final_model"
  local dataset_dir="${DATA_ROOT}/${target}"
  local output_dir
  local dependency_option
  local -a sbatch_args=()

  if [[ ! -d "${final_model}" ]]; then
    echo "missing-model"
    return
  fi
  if [[ -f "${run_dir}/${LOW_REWARD_STOP_MARKER}" ]]; then
    echo "skipped-stop-marker"
    return
  fi

  output_dir="${final_model}/hold_out_eval"
  if [[ -f "${output_dir}/metrics.csv" && -f "${output_dir}/summary.csv" ]]; then
    echo "done"
    return
  fi
  if [[ -e "${output_dir}" ]]; then
    echo "partial"
    return
  fi

  dependency_option="$(dependency_arg "${serial_dependency}")"
  if [[ -n "${dependency_option}" ]]; then
    sbatch_args+=("${dependency_option}")
  fi

  sbatch --parsable \
    "${sbatch_args[@]}" \
    --job-name="holdout-sft-warmup" \
    --output="logs/holdout-sft-warmup-%j.log" \
    --gres=gpu:1 \
    --time="01:30:00" \
    --partition="sc-gpu" \
    --wrap="cd ${REPO_DIR} && if [ -f '${run_dir}/${LOW_REWARD_STOP_MARKER}' ]; then echo 'Skipping hold-out because stop marker exists: ${run_dir}/${LOW_REWARD_STOP_MARKER}'; cat '${run_dir}/${LOW_REWARD_STOP_MARKER}'; exit 0; fi && source \$HOME/anaconda3/etc/profile.d/conda.sh && conda activate py312_cu118 && python eval/hold_out_styles/generate-and-analyze-completions.py concept='${target}' model_name_or_path='${final_model}' output_dir='${output_dir}' paths.storage_root='${STORAGE_ROOT}'"
}

previous_holdout_job_id=""

for original_model in "${original_models[@]}"; do
  echo "Original model | ${original_model}"

  for target in "${targets[@]}"; do
    holdout_job_id="$(submit_warmup_holdout "${original_model}" "${target}" "${previous_holdout_job_id}")"
    echo "R2 warm-up hold-out | ${target}"

    if [[ "${holdout_job_id}" == "done" ]]; then
      echo "  holdout: already exists"
    elif [[ "${holdout_job_id}" == "missing-model" ]]; then
      echo "  holdout: missing warm-up final_model"
    elif [[ "${holdout_job_id}" == "skipped-stop-marker" ]]; then
      echo "  holdout: skipped because stop marker exists"
    elif [[ "${holdout_job_id}" == "partial" ]]; then
      echo "  holdout: incomplete existing output"
    else
      echo "  holdout: ${holdout_job_id}"
      previous_holdout_job_id="${holdout_job_id}"
    fi
  done
done
