# Completions Analysis

Utilities for inspecting model completions logged by W&B training runs.

The main script, `analyze-completions.py`, looks for completions that
mention any refusal-pattern entity for the selected concept exactly or through
the fuzzy forgetting reward. For `CONCEPT=Rihanna`, that includes `Rihanna`,
`Fenty Beauty`, `Umbrella`, and the rest of `REFUSAL_PATTERNS["Rihanna"]`.

## What It Scans

The script downloads `media/table/completions_*.table.json` from W&B training
runs whose config has
`hydra.experiment.forget_concept == --concept`, then scans those downloaded
tables.
When `--model-name` is set, runs are also filtered by `hydra.model.name`.
For each scanned completion, the script also computes `fuzzy_reward` using the
same fuzzy forgetting reward logic from `src.reward.fuzzy`.

## What It Reports

- Exact target-entity matches, for example `Rihanna` or `Fenty Beauty`
- Fuzzy reward matches, where `fuzzy_reward == 0.0`
- Fuzzy reward matches without an exact target-entity match
- Exact matches for any configured reward-list entity
- A compact CSV containing matching row metadata
- A summary CSV containing aggregate leakage counts

The most important signal for fuzzy-only leakage is:

```text
exact_target == False
fuzzy_reward == 0.0
exact_entities == ""
```

That means the completion does not contain an exact target-entity match, but it
still matches the fuzzy forgetting reward logic.

## Example

```bash
python eval/hold_out_styles/analyze-completions.py \
  --concept Confucius \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --output-csv outputs/completion_analysis/confucius-wandb-completions.csv
```

The script reads `WANDB_PROJECT` from the repository `.env` file or
environment. `WANDB_ENTITY` is optional and can also be read from
`.env`/environment.

When `--output-csv` is set, a second CSV is written next to it using the
`.summary.csv` suffix. For example:

```text
outputs/completion_analysis/confucius-wandb-completions.csv
outputs/completion_analysis/confucius-wandb-completions.summary.csv
```

The summary CSV includes one `all` row per source kind, aggregate rows, and one
row per W&B run/source-kind pair. It reports:

- `total_rows`: all rows read from that source kind before filtering
- `training_mode`: the training label from the W&B run config, or `all`
- `matching_filter_rows`: rows analyzed after loading completion text
- `near_target_rows`: rows where the fuzzy reward would be `0.0`
- `near_without_exact_rows`: fuzzy reward matches without an exact match
- `exact_entity_rows`
- `near_without_exact_rate`
- `near_without_exact_rate_std`: sample standard deviation of per-run `near_without_exact_rate` values for aggregate rows

For Slurm usage, use:

```bash
CONCEPT=Confucius \
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct \
OUTPUT_CSV=outputs/completion_analysis/confucius-wandb-completions.csv \
sbatch scripts/completitions_analysis/slurm-analyze-completions.sh
```
