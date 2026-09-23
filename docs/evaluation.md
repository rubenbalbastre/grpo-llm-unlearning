# Evaluation

The repo has two evaluation paths:

- RWKU benchmark evaluation under `eval/rwku/`
- hold-out completion generation and analysis under `eval/behaviour/`

## RWKU

The main RWKU config is `config/eval_rwku.yaml`.

Typical command:

```bash
python -m eval.rwku.rwku \
  evaluation.model_name_or_path=outputs/<run>/final_model \
  evaluation.output_dir=outputs/<run>/final_model/eval_rwku \
  evaluation.subjects="Stephen King"
```

Slurm:

```bash
sbatch scripts/run-eval-rwku.sh
```

To evaluate the final model from a run directory:

```bash
CHECKPOINT_ROOT=outputs/<run> sbatch scripts/run-eval-rwku.sh
```

## RWKU Metrics

Forget, neighbor, and MIA sets are evaluated after optional subject filtering.
Utility can be capped with `evaluation.limits.max_utility_examples`.

Implemented metrics:

- generation: `rouge_l_recall`
- MIA: `loss`

Other MIA metrics in the config are placeholders and currently raise
`NotImplementedError` if enabled.

## W&B Linkage

With `evaluation.wandb.link_to_training_run=true`, evaluation reads
`wandb_run.json` from the training run directory and resumes the same W&B run.
Evaluation metrics are logged as `rwku/*`, and model/result artifacts can be
logged depending on `evaluation.wandb`.

## Hold-Out Completion Analysis

This evaluation generates completions for the GRPO hold-out prompts of one
forget concept and scores them with the LLM-judge rubrics.

Typical command:

```bash
python eval/behaviour/generate-and-analyze-completions.py \
  model_name_or_path=outputs/<run>/final_model \
  concept="Stephen King" \
  output_dir=outputs/<run>/final_model/hold_out_eval
```

Useful overrides:

```bash
python eval/behaviour/generate-and-analyze-completions.py \
  model_name_or_path=outputs/<run>/final_model \
  concept="Stephen King" \
  max_examples=64 \
  judge_concurrency=8
```

The script reads `.env`, so `OPENAI_API_KEY` is required for the judge. Outputs
are written to:

```text
outputs/<run>/final_model/hold_out_eval/
  metrics.csv
  summary.csv
```

`metrics.csv` contains one row per prompt/completion. `summary.csv` contains
the average rubric scores for that checkpoint and concept.

## Final Tables

Create the RWKU and behavioural article tables from completed evaluations:

```bash
scripts/tables/final-table.sh
```

Four CSV files are written under `outputs/tables/`: author-level and aggregate
tables for RWKU and behavioural evaluation. They are also packaged as
`outputs/tables/final-tables.tar.gz`.

## Training-Dynamics Hold-Out Analysis

This script evaluates completions logged during training. It downloads only the
selected recent W&B completion tables, scores prompt/completion pairs with the
same hold-out rubrics, and aggregates in this order:

```text
generation completions -> prompts -> runs -> authors -> final group
```

The final group is `[model_size, reward_type, variant]`, where `variant` is the
training variant such as `original` or `warmed`.

First download or reuse the selected W&B tables and inspect the inventory:

```bash
python eval/behaviour/analyze-training-dynamics.py \
  --download-only \
  --last-steps 5
```

Then score and aggregate using the local downloads:

```bash
python eval/behaviour/analyze-training-dynamics.py \
  --skip-download \
  --last-steps 5 \
  --judge-concurrency 8
```

To recompute summaries from existing per-run rubric CSVs without making OpenAI
API calls:

```bash
python eval/behaviour/analyze-training-dynamics.py \
  --skip-download \
  --reaverage-cached-only \
  --last-steps 5
```

The script reads `.env`. `WANDB_PROJECT` is required when downloading from W&B,
and `OPENAI_API_KEY` is required when scoring uncached completions.

Default filters are intentionally hard-coded in
`eval/behaviour/analyze-training-dynamics.py`: W&B job type `training`,
notes `lluis-vives-runs-1-1M`, run names containing `original` or `rN-warmed`,
and the current author subset. Runs with fewer than 101 completion table files
are skipped.

Outputs are written under `outputs/training_dynamics_hold_out_rubrics/`:

```text
download_inventory.csv
download_inventory_by_author_reward.csv
metrics.csv
prompt_summary.csv
run_summary.csv
author_summary.csv
summary.csv
summary_median_iqr.csv
run_metrics/*.csv
wandb-downloads/
```

Use `summary.csv` for mean scores by `[model_size, reward_type, variant]`. Use
`summary_median_iqr.csv` for median, Q1, and Q3 by the same group. Per-run judge
results are cached in `run_metrics/`; pass `--overwrite-run-metrics` only when
you want to rescore them.
