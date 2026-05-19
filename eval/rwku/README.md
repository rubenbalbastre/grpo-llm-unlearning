# RWKU Evaluation

This folder contains the RWKU evaluation scripts. The entrypoint is:

```bash
python eval/rwku/rwku.py
```

| Set | Ability | RWKU dataset | Metric computed |
| --- | --- | --- | --- |
| Forget Set | Knowledge Memorization | Forget FB | ROUGE-L recall |
| Forget Set | Knowledge Manipulation | Forget QA | ROUGE-L recall |
| Forget Set | Adversarial Attack | Forget AA | ROUGE-L recall |
| Neighbor Set | Knowledge Memorization | Neighbor FB | ROUGE-L recall |
| Neighbor Set | Knowledge Manipulation | Neighbor QA | ROUGE-L recall |
| MIA Set | Knowledge Memorization | FM | Loss by default; configurable via boolean MIA options |
| MIA Set | Knowledge Memorization | RM | Loss by default; configurable via boolean MIA options |
| Utility Set | General Ability | MMLU | Accuracy via answer-option likelihood |
| Utility Set | Reasoning Ability | BBH | Exact match |
| Utility Set | Truthfulness | TruthfulQA | Accuracy via answer-option likelihood |
| Utility Set | Factuality | TriviaQA | Token F1 |
| Utility Set | Fluency | AlpacaEval | Weighted bi-/tri-gram entropy |

By default, only the MIA `loss` metric is computed:

```bash
python eval/rwku/rwku.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir outputs/eval_rwku/qwen_stephen_king \
  --subjects "Stephen King" \
  --compute_mia_loss
```

The four paper metrics are represented by:

```text
--compute_mia_loss
--compute_mia_zlib
--compute_mia_min_k
--compute_mia_min_k_plus_plus
```

Each boolean option can be disabled with the matching `--no-...` flag. Currently only `Loss` is implemented. The other MIA options are present in the interface and raise `NotImplementedError` if enabled.

Set-level execution can be toggled independently:

```bash
python eval/rwku/rwku.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir outputs/eval_rwku/qwen_forget_only \
  --subjects "Stephen King" \
  --run_forget_set \
  --no-run_neighbor_set \
  --no-run_mia_set \
  --no-run_utility_set
```
