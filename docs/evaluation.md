# Evaluation

The repo has two evaluation paths:

- RWKU benchmark evaluation under `eval/rwku/`
- hold-out completion generation and analysis under `eval/hold_out_styles/`

## RWKU

The main RWKU config is `config/eval.yaml`.

Typical command:

```bash
python -m eval.rwku.rwku \
  evaluation.model_name_or_path=outputs/<run>/checkpoint-175 \
  evaluation.output_dir=outputs/<run>/checkpoint-175/eval_rwku \
  evaluation.subjects="Stephen King"
```

Slurm:

```bash
sbatch scripts/eval-rwku.sh
```

To evaluate every checkpoint in a run directory:

```bash
CHECKPOINT_ROOT=outputs/<run> sbatch scripts/eval-rwku.sh
```

Set `INCLUDE_FINAL_MODEL=true` to include `final_model`.

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

Generate completions and analyze them:

```bash
python eval/hold_out_styles/generate-and-analyze-completions.py \
  model_name_or_path=outputs/<run>/checkpoint-175 \
  concept="Stephen King" \
  output_dir=outputs/<run>/checkpoint-175/hold_out_styles
```

Analyze W&B completion tables:

```bash
python eval/hold_out_styles/analyze-completions.py \
  --concept "Stephen King" \
  --reward-type r2 \
  --output-csv outputs/completion_analysis/stephen-king-r2
```

The summary groups by `reward_type` and run name.
