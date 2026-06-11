# RWKU Evaluation

RWKU evaluation is configured through `config/eval.yaml` or checkpoint overrides
passed by `scripts/slurm-eval-rwku.sh`.
If authentication is needed, it is read from `HUGGINGFACE_HUB_TOKEN` in the
repository `.env` file.

Select a saved training run and an optional target filter in the config:

```yaml
evaluation:
  model_name_or_path: outputs/<wandb_run_name>/final_model
  output_dir: outputs/<wandb_run_name>/eval_rwku
  subjects: Stephen King
```

Configure evaluation work in the same file:

```yaml
evaluation:
  sets:
    forget: true
    neighbor: true
    mia: true
    utility: false
  metrics:
    generation: rouge_l_recall
    mia:
      loss: true
      zlib: false
      min_k: false
      min_k_plus_plus: false
  limits:
    max_utility_examples: 2000
    utility_sample_strategy: random
    utility_sample_seed: 42
  wandb:
    enabled: true
    run_name: rwku-${evaluation.subjects}
    artifact_name: rwku-stephen-king-model
    log_model_artifact: true
    link_to_training_run: true
```

Forget, neighbor, and MIA sets are always evaluated in full after optional
subject filtering. Utility can be capped with `max_utility_examples`;
`utility_sample_strategy: random` selects a fixed-seed random subset before
sharding. Use `first` only when you intentionally want the first utility rows
from the Hugging Face dataset.

When W&B logging is enabled, `WANDB_API_KEY` and `WANDB_PROJECT` are read from
`.env`. With `link_to_training_run: true`, evaluation resumes the training W&B
run recorded in
`final_model/wandb_run.json` and logs scalar `rwku/*` metrics there. With
`log_model_artifact: true`, it also uploads the local model directory and
result files as a W&B `model` artifact. Set `link_to_training_run: false` for
models without that recorded training-run identity. For a linked run,
`WANDB_PROJECT` must match the project recorded during training.

| Set | Ability | RWKU dataset | Metric computed |
| --- | --- | --- | --- |
| Forget Set | Knowledge Memorization | Forget FB | ROUGE-L recall |
| Forget Set | Knowledge Manipulation | Forget QA | ROUGE-L recall |
| Forget Set | Adversarial Attack | Forget AA | ROUGE-L recall |
| Neighbor Set | Knowledge Memorization | Neighbor FB | ROUGE-L recall |
| Neighbor Set | Knowledge Manipulation | Neighbor QA | ROUGE-L recall |
| MIA Set | Knowledge Memorization | FM | Loss |
| MIA Set | Knowledge Memorization | RM | Loss |
| Utility Set | General Ability | MMLU | Accuracy via answer-option likelihood |
| Utility Set | Reasoning Ability | BBH | Exact match |
| Utility Set | Truthfulness | TruthfulQA | Accuracy via answer-option likelihood |
| Utility Set | Factuality | TriviaQA | Token F1 |
| Utility Set | Fluency | AlpacaEval | Weighted bi-/tri-gram entropy |

Only the MIA `loss` metric is implemented currently. Enabling `zlib`, `min_k`,
or `min_k_plus_plus` raises `NotImplementedError`.

For MIA, per-example `loss` in `rwku_mia.jsonl` is mean token negative log
likelihood. The summary table sums that normalized loss within each subject
and averages subject sums, keeping the RWKU single-target/all-target scale.
