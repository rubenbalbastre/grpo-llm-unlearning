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

## Data Licenses

The repository code is Apache-2.0 licensed. Dataset contents and derived data
artifacts keep the license terms of their source data.

The Hugging Face dataset exported by `scripts/dataset/upload-data-to-hf.py` is
declared as CC BY 4.0 and attributed to RWKU (`jinzhuoran/RWKU`). It is a
processed/modified version of RWKU: rows are filtered by forget concept,
prompts are normalized, columns are renamed, splits are reorganized for this
project, the reference `completion` column is removed from GRPO train and
holdout exports, and a `concept` column is added during export.

PURGE material is third-party data from the PURGE project and is MIT-licensed by
its original authors. It is attributed separately from the Apache-2.0 code in
this repository.

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

## Hugging Face Dataset Repository

Export all concept folders under `data/` into one private Hugging Face dataset
repo:

```bash
python scripts/dataset/upload-data-to-hf.py
```

Authentication is read from `.env` using `HF_TOKEN` or
`HUGGINGFACE_HUB_TOKEN`.

Override the data root or repo id if needed:

```bash
python scripts/dataset/upload-data-to-hf.py data username/unlearning-data
```

The parquet files include a `concept` column.
`grpo_train.parquet` and `holdout.parquet` are prompt-only for the model-facing
task and do not include the RWKU reference `completion` column. SFT parquet
files keep `completion`.

The repository files are:

```text
grpo_train.parquet
sft_train.parquet
sft_validation.parquet
holdout.parquet
README.md
```

Split mapping:

```text
grpo/train -> grpo_train.parquet
sft/train -> sft_train.parquet
sft/test -> sft_validation.parquet
grpo/test -> holdout.parquet
```

The uploaded `README.md` declares these files under `configs.data_files` so the
Hugging Face Dataset Viewer can discover the custom split names. It also
includes the RWKU attribution, CC BY 4.0 dataset license, processing notice, and
PURGE MIT third-party data attribution.

Download the Hugging Face dataset back into the local `data/<concept>` layout:

```bash
python scripts/dataset/download-data-from-hf.py
```

Override the repo id or output directory if needed:

```bash
python scripts/dataset/download-data-from-hf.py username/unlearning-data data
```
