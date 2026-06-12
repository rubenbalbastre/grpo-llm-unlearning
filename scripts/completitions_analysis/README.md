# Completions Analysis

Utilities for inspecting model completions after unlearning runs.

The main script, `analyze-adaptive-completions.py`, looks for completions that
mention a target concept exactly or through near-spellings. The motivating case
is `Rihanna` vs variants such as `Rihana`, `Rhianna`, or `Rihannna`.

## What It Scans

For each run directory passed to the script, it reads:

- `completions/*.parquet` from TRL completion logging
- `checkpoint-*/eval_rwku/rwku_generations.jsonl`

When `--forgetting-reward 1` is used, only TRL completion rows with
`forgetting_reward == 1.0` are analyzed. RWKU rows are ignored in that mode,
because their `reward`-like value is `rouge_l_recall`, not the training
forgetting reward.

## What It Reports

- Exact target-name matches, for example `Rihanna`
- Near target-name variants, for example `Rihana` or `Rihannna`
- Near variants without an exact target-name match
- Exact matches for any configured reward-list entity
- A CSV containing the matching rows
- A summary CSV containing aggregate leakage counts

The most important signal for reward evasion is:

```text
forgetting_reward == 1.0
exact_target == False
target_variant != ""
exact_entities == ""
```

That means the reward considered the completion successful, but the completion
still contains a near-spelling of the forgotten entity.

## Current Run Groups

Standard runs:

```text
3845
3857
3859
3861
3863
```

Adaptive runs:

```text
3855
3868
3870
3874
3886
```

`38744` was also mentioned during analysis, but
`outputs/unlearning-38744` does not exist locally.

## Example

```bash
scripts/completitions_analysis/analyze-adaptive-completions.py \
  outputs/unlearning-3855 outputs/unlearning-3868 \
  --concept Rihanna \
  --forgetting-reward 1 \
  --output-csv outputs/completion_analysis/rihanna-adaptive-runs-forgetting-reward-1.csv
```

When `--output-csv` is set, a second CSV is written next to it using the
`.summary.csv` suffix. For example:

```text
outputs/completion_analysis/rihanna-adaptive-runs-forgetting-reward-1.csv
outputs/completion_analysis/rihanna-adaptive-runs-forgetting-reward-1.summary.csv
```

The summary CSV includes one `all` row per source kind and one row per
`unlearning-*` run/source-kind pair. It reports:

- `total_rows`: all rows read from that source kind before filtering
- `matching_filter_rows`: rows kept after filters such as `--forgetting-reward 1`
- `exact_target_rows`
- `near_target_rows`
- `near_without_exact_rows`
- `exact_entity_rows`
- `near_without_exact_rate`

For Slurm usage, use:

```bash
RUNS="3855 3868 3886 3874 3870" \
OUTPUT_CSV=outputs/completion_analysis/rihanna-adaptive-runs-forgetting-reward-1.csv \
sbatch scripts/completitions_analysis/slurm-analyze-completions.sh
```
