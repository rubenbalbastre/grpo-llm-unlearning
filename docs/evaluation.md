# Evaluation

The repository has two final-model evaluation paths:

- RWKU benchmark evaluation under `eval/rwku/`
- behavioural hold-out generation and LLM-judge scoring under `eval/behaviour/`

The matrix runner submits both by default. Set `RUN_RWKU_EVAL` or
`RUN_HOLD_OUT_EVAL` in `scripts/run-target-reward-matrix.py` to disable one.
Terminal training rollouts are scored separately from W&B after training.

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

## Terminal Training Behaviour

`eval/behaviour/prepare_training_rollouts.py` filters the configured W&B runs,
skips runs with fewer than 101 completion tables, downloads only the most recent
selected tables, and scores unique prompt/completion pairs. Judge defaults come
from `config/eval_behaviour.yaml`.

Download and score the last five optimizer steps:

```bash
scripts/run-training-behavior.sh
```

Reuse existing downloads and score only missing pairs:

```bash
scripts/run-training-behavior.sh --skip-download
```

Per-run judge results are saved incrementally under
`outputs/terminal_training_audit/run_metrics/`. The downloaded tables and run
inventory are stored in the same directory.

## Final Tables

After all required evaluations and terminal training rollout scores exist, run:

```bash
scripts/tables/final-table.sh
```

`scripts/tables/final-table.sh` runs
`eval/behaviour/create_final_terminal_training_audit.py` to aggregate those
cached scores. It does not download rollouts or call the judge. The resulting
table directory and archive contain:

```text
outputs/tables/rwku.csv
outputs/tables/rwku_authors.csv
outputs/tables/heldout_behaviour.csv
outputs/tables/heldout_behaviour_authors.csv
outputs/tables/terminal_training_behaviour.csv
outputs/tables/terminal_training_behaviour_authors.csv
outputs/tables/final-tables.tar.gz
```

The terminal-training author table averages generations, prompts, and optimizer
steps into one row per author and condition. Aggregate tables report median, Q1,
and Q3 across author rows, grouped by reward function, model size, and training
initialization, plus the author count. Runs marked with `low_reward_stop.json`
are excluded from the final-model tables.

RWKU forget, neighbor, and MIA values are deltas from the matching original
model baseline. Utility values remain raw. The held-out behaviour table reads
existing `summary.csv` files and does not rescore `metrics.csv`.

`WANDB_PROJECT` is required for downloading. `OPENAI_API_KEY` is required only
when uncached prompt/completion pairs must be judged.
