# Data

The training data is derived from the RWKU dataset configured under
`standard_data` in `config/train.yaml`. Each generated dataset is scoped to one
forget concept and stored with Hugging Face `DatasetDict.save_to_disk`.

## Split Generation

For the selected concept, `generate_sft_grpo_splits.py`:

1. Filters RWKU rows by `subject`.
2. Normalizes each instruction to its question portion.
3. Retains `prompt`, reference `completion`, and `subject`.
4. Applies the configured dataset-size limit and deterministic shuffle.
5. Separates the GRPO training set, GRPO hold-out set, and SFT subset.
6. Uses OpenAI to generate `broad_completion` for the SFT subset.

Generate one target locally or through Slurm:

```bash
python generate_sft_grpo_splits.py experiment.forget_concept="Stephen King"
sbatch scripts/generate-data-splits.sh experiment.forget_concept="Stephen King"
```

Broad-completion generation requires `OPENAI_API_KEY`. Concurrency and retry
limits can be changed with `OPENAI_CONCURRENCY` and `OPENAI_MAX_RETRIES`.

## Local Layout

Data is written under `data/<concept>/` by default:

```text
data/<concept>/
  dataset_dict.json
  grpo/train/
  grpo/test/
  sft/train/
  sft/test/
```

GRPO reads `grpo/train`; behavioural evaluation uses `grpo/test`. SFT reads the
two SFT splits and selects either the RWKU `completion` or generated
`broad_completion` according to `training.sft.objective`.

The matrix runner requires all four splits for every target in its `TARGETS`
list before it submits any jobs.

## Hugging Face Transfer

Upload every concept directory under `data/` to one private Hugging Face
dataset repository:

```bash
python scripts/dataset/upload-data-to-hf.py
```

The default repository name is `grpo-unlearning-data`. A bare name is resolved
under the authenticated account. Override the data root or repository ID with:

```bash
python scripts/dataset/upload-data-to-hf.py data username/repository
```

The upload always includes all concept directories and creates:

```text
grpo_train.parquet
sft_train.parquet
sft_validation.parquet
holdout.parquet
README.md
```

Every parquet file includes a `concept` column. The GRPO train and hold-out
exports omit the RWKU reference `completion`; both SFT exports retain it. The
generated dataset card exposes separate `grpo` and `sft` configurations so
their different schemas load correctly and render in the Dataset Viewer.

Download the private repository back into the local per-concept layout:

```bash
python scripts/dataset/download-data-from-hf.py
python scripts/dataset/download-data-from-hf.py username/repository data
```

Both transfer scripts load `.env` and require `HF_TOKEN`.

## Licenses

Repository code and documentation are Apache-2.0 licensed. The exported dataset
is declared CC BY 4.0 with attribution to RWKU (`jinzhuoran/RWKU`) and a clear
notice that it is a processed/modified derivative.

PURGE material, when present, remains MIT-licensed third-party data attributed
to the PURGE project. It is not covered by this repository's Apache-2.0 code
license.
