# RWKU Evaluation

RWKU evaluation is configured through `config/eval.yaml` or checkpoint overrides
passed by `scripts/eval-rwku.sh`.
If authentication is needed, it is read from `HUGGINGFACE_HUB_TOKEN` in the
repository `.env` file.

Select a saved model directory and an optional target filter in the config.
For checkpoint-only runs, point this at one `checkpoint-*` directory:

```yaml
evaluation:
  model_name_or_path: outputs/<wandb_run_name>/checkpoint-175
  output_dir: outputs/<wandb_run_name>/checkpoint-175/eval_rwku
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

Forget, neighbor, MIA, and utility sets are evaluated after optional subject
filtering. Utility can be capped with `max_utility_examples`;
`utility_sample_strategy: random` selects a fixed-seed random subset after
subject filtering and before sharding. Use `first` only when you intentionally
want the first utility rows from the Hugging Face dataset. All utility metrics
follow the released RWKU evaluator: MMLU uses its five-shot prompt and
next-token probability over A-D; BBH uses few-shot chain-of-thought and
normalized extracted-answer exact match; TruthfulQA uses the QA primer and MC1
from summed completion log probabilities; TriviaQA uses its five-shot prompt
and maximum token F1 over answer aliases; and fluency uses NLTK-tokenized,
RWKU-weighted bigram/trigram entropy.

When W&B logging is enabled, `WANDB_API_KEY` and `WANDB_PROJECT` are read from
`.env`. With `link_to_training_run: true`, evaluation resumes the training W&B
run recorded in `outputs/<wandb_run_name>/wandb_run.json` and logs scalar
`rwku/*` metrics there. For `final_model` or `checkpoint-*` eval paths, RWKU
derives the run directory from the parent folder. With `log_model_artifact:
true`, it also uploads the local model directory and result files as a W&B
`model` artifact. Set `link_to_training_run: false` for models without that
recorded training-run identity. For a linked run, `WANDB_PROJECT` must match
the project recorded during training.grpo.

| Set | Ability | RWKU dataset | Metric computed |
| --- | --- | --- | --- |
| Forget Set | Knowledge Memorization | Forget FB | ROUGE-L recall |
| Forget Set | Knowledge Manipulation | Forget QA | ROUGE-L recall |
| Forget Set | Adversarial Attack | Forget AA | ROUGE-L recall |
| Neighbor Set | Knowledge Memorization | Neighbor FB | ROUGE-L recall |
| Neighbor Set | Knowledge Manipulation | Neighbor QA | ROUGE-L recall |
| MIA Set | Knowledge Memorization | FM | Loss |
| MIA Set | Knowledge Memorization | RM | Loss |
| Utility Set | General Ability | MMLU | Next-token A-D accuracy |
| Utility Set | Reasoning Ability | BBH | Exact match |
| Utility Set | Truthfulness | TruthfulQA | MC1 summed-log-probability accuracy |
| Utility Set | Factuality | TriviaQA | Maximum token F1 over answer aliases |
| Utility Set | Fluency | AlpacaEval | Official weighted bi-/tri-gram entropy |

Only the MIA `loss` metric is implemented currently. Enabling `zlib`, `min_k`,
or `min_k_plus_plus` raises `NotImplementedError`.

For MIA, per-example `loss` in `rwku_mia.jsonl` is mean token negative log
likelihood. The summary table sums that normalized loss within each subject
and averages subject sums, keeping the RWKU single-target/all-target scale.
