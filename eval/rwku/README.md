# RWKU Evaluation

RWKU evaluation is configured only through `config/eval.yaml`. The training
Slurm job runs evaluation automatically after saving its final model, and
`scripts/slurm-eval-rwku.sh` runs the same configured evaluation independently.
If authentication is needed, it is read from `HUGGINGFACE_HUB_TOKEN` in the
repository `.env` file.

Set the model, results directory, and optional target filter in the config:

```yaml
evaluation:
  model_name_or_path: outputs/adaptive_grpo/final_model
  output_dir: outputs/adaptive_grpo/eval_rwku
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
    max_examples: null
    max_mia_examples: null
    max_utility_examples: null
  wandb:
    enabled: true
    project: machine-unlearning-llm
    run_name: rwku-${evaluation.subjects}
    artifact_name: rwku-stephen-king-model
    log_model_artifact: true
    link_to_training_run: true
```

When W&B logging is enabled and `WANDB_API_KEY` is present in `.env`,
`link_to_training_run: true` resumes the training W&B run recorded in
`final_model/wandb_run.json` and logs scalar `rwku/*` metrics there. With
`log_model_artifact: true`, it also uploads the local model directory and
result files as a W&B `model` artifact. Set `link_to_training_run: false` for
models without that recorded training-run identity.

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
