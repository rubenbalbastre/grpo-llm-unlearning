# Evaluation

The repository has two final-model evaluation paths:

- RWKU benchmark evaluation under `eval/rwku/`
- behavioural hold-out generation and LLM-judge scoring under `eval/behaviour/`

The matrix runner submits both by default. Set `RUN_RWKU_EVAL` or
`RUN_HOLD_OUT_EVAL` in `scripts/run-target-reward-matrix.py` to disable one.

## RWKU

Evaluate a completed training run with:

```bash
CHECKPOINT_ROOT=outputs/<run> sbatch scripts/run-eval-rwku.sh
```

The launcher reads the subject from `outputs/<run>/hydra_config.yaml`, evaluates
`outputs/<run>/final_model`, and writes results to:

```text
outputs/<run>/final_model/eval_rwku/
```

With multiple allocated GPUs, the launcher runs one evaluation shard per GPU
and merges them. It does not use Accelerate because each process evaluates an
independent data shard.

For a direct baseline or local run, provide Hydra overrides:

```bash
python eval/rwku/rwku.py \
  evaluation.model_name_or_path=Qwen/Qwen2.5-3B-Instruct \
  evaluation.output_dir=outputs/example/eval_rwku \
  evaluation.subjects="Stephen King"
```

RWKU evaluates forget, neighbor, MIA, and utility sets according to
`config/eval_rwku.yaml`. Generation uses ROUGE-L recall and the implemented MIA
metric is loss. Enabling the unimplemented MIA options raises
`NotImplementedError`.

## Behavioural Hold-Out Evaluation

This path generates completions for `grpo/test` and independently scores:

- lexical leakage
- semantic leakage
- prompt helpfulness
- broad-topic helpfulness
- refusal
- language drift

Run it on a final model:

```bash
sbatch scripts/run-eval-behaviour.sh \
  concept="Stephen King" \
  model_name_or_path=outputs/<run>/final_model \
  output_dir=outputs/<run>/final_model/hold_out_eval
```

The judge model, reasoning effort, and concurrency default to
`config/eval_behaviour.yaml` and can be overridden on the command line. The
judge requires `OPENAI_API_KEY`.

Outputs are:

```text
outputs/<run>/final_model/hold_out_eval/
  metrics.csv                # one row per prompt/completion
  summary.csv                # mean of each rubric
```

## Final Tables

After all required evaluations have produced their summaries, run:

```bash
scripts/tables/final-table.sh
```

This creates:

```text
outputs/tables/rwku.csv
outputs/tables/rwku_authors.csv
outputs/tables/behaviour.csv
outputs/tables/behaviour_authors.csv
outputs/tables/final-tables.tar.gz
```

The aggregate CSVs report median, Q1, and Q3 grouped by reward function, model
size, and training initialization, plus the author count. Runs marked with
`low_reward_stop.json` are excluded.

RWKU forget, neighbor, and MIA values are expressed as deltas from the matching
original-model baseline for each model size and author. SFT warmup rows are also
matched to those baselines and reported as `r2-warmup` when their RWKU results
are present. Utility values remain the evaluated model's raw values.

The behavioural table reads existing `summary.csv` files only. It does not
rescore `metrics.csv` or reconstruct missing summaries.

## Training Dynamics

`eval/behaviour/analyze-training-dynamics.py` evaluates completions logged in
W&B during GRPO training. It filters the configured author set and qualifying
training runs, skips runs with fewer than 101 completion tables, and downloads
only the most recent selected tables.

Download data and inspect the inventory first:

```bash
python eval/behaviour/analyze-training-dynamics.py \
  --download-only \
  --last-steps 5
```

Then reuse those downloads for scoring and aggregation:

```bash
python eval/behaviour/analyze-training-dynamics.py \
  --skip-download \
  --last-steps 5
```

Per-run judge results are saved incrementally under
`outputs/training_dynamics_hold_out_rubrics/run_metrics/`. Final outputs include
prompt, run, author, and experiment-group summaries. `summary_median_iqr.csv`
contains median, Q1, and Q3 grouped by model size, reward type, and training
variant.

`WANDB_PROJECT` is required for downloading. `OPENAI_API_KEY` is required only
when uncached prompt/completion pairs must be judged.
