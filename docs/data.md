# Data

The training pipeline uses RWKU-derived splits saved with Hugging Face
`DatasetDict.save_to_disk`.

## Source

The source dataset is configured in `config/train.yaml`:

```yaml
standard_data:
  name: jinzhuoran/RWKU
  config_name: train_refusal_phi3
  split: train
  subject_column: subject
  prompt_column: instruction
  dataset_size: 300
  shuffle_seed: 42
  grpo_test_size: 0.2
  sft_train_size: 0.10
  sft_test_size: 0.05
```

`generate_sft_grpo_splits.py` filters rows by `experiment.forget_concept`,
normalizes each instruction to the question portion, removes empty prompts, and
renames `instruction` to `prompt` and `output` to `completion`.

## Broad Completions

The SFT subset also gets a `broad_completion` column generated with an OpenAI
model. This is used by `sft_warm_up.py` when `training.sft.objective=broad`.

Relevant environment variables:

```bash
OPENAI_API_KEY=...
OPENAI_CONCURRENCY=4
OPENAI_MAX_RETRIES=8
```

## Generate Splits

Local:

```bash
python generate_sft_grpo_splits.py experiment.forget_concept="Stephen King"
```

Slurm:

```bash
sbatch scripts/generate-data-splits.sh experiment.forget_concept="Stephen King"
```

## Output Layout

Splits are saved under the path produced by
`src.data_preprocessing.utils.build_dataset_path`, scoped by
`paths.storage_root` and the forget concept.

The saved dataset contains:

```text
grpo/train
grpo/test
sft/train
sft/test
```

Training loads those splits through `src.data_preprocessing.load_dataset`.
