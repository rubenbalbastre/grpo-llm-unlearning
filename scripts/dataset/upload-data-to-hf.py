#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_from_disk
from dotenv import load_dotenv
from huggingface_hub import HfApi


SPLITS = {
    "grpo/train": "grpo_train.parquet",
    "sft/train": "sft_train.parquet",
    "sft/test": "sft_validation.parquet",
    "grpo/test": "holdout.parquet",
}
GRPO_SPLITS = {"grpo/train", "grpo/test"}


def write_readme(
    export_dir: Path,
    repo_id: str,
    data_dir: Path,
    dataset: DatasetDict,
    concepts: list[str],
) -> None:
    rows = "\n".join(
        f"| `{filename}` | `{split}` | {len(dataset[split])} |"
        for split, filename in SPLITS.items()
    )
    concept_text = "\n".join(f"- {concept}" for concept in concepts)
    readme = f"""---
task_categories:
- text-generation
license: cc-by-4.0
configs:
- config_name: default
  data_files:
  - split: grpo_train
    path: grpo_train.parquet
  - split: sft_train
    path: sft_train.parquet
  - split: sft_validation
    path: sft_validation.parquet
  - split: holdout
    path: holdout.parquet
---

# {repo_id}

Dataset splits for targeted machine unlearning experiments.

## License and Attribution

This dataset is released as CC BY 4.0.

The data is derived from RWKU (`jinzhuoran/RWKU`) and should be attributed to
the RWKU authors. These files are a processed/modified version of RWKU: rows
were filtered by forget concept, prompts were normalized, columns were renamed,
splits were reorganized for this project, the reference `completion` column was
removed from GRPO train and holdout splits, and a `concept` column was added
during export.

If PURGE material is present in the source data, it is third-party material
attributed to the PURGE project and is MIT-licensed by its original authors.

The source code used to create these files is licensed separately under
Apache-2.0 in the `machine-unlearning-llm` repository.

Source dataset directory: `{data_dir}`

Concepts:

{concept_text}

| File | Source split | Rows |
| --- | --- | ---: |
{rows}
"""
    (export_dir / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs="?", type=Path, default=Path("data"), help="Path to data/.")
    parser.add_argument("repo_id", nargs="?", default="grpo-unlearning-data", help="Hugging Face dataset repo.")
    return parser.parse_args()


def add_concept_column(dataset: DatasetDict, concept: str) -> DatasetDict:
    splits = {}
    for split in SPLITS:
        split_dataset = dataset[split]
        if split in GRPO_SPLITS and "completion" in split_dataset.column_names:
            split_dataset = split_dataset.remove_columns("completion")
        splits[split] = split_dataset.add_column(
            "concept",
            [concept] * len(split_dataset),
        )
    return DatasetDict(splits)


def load_concept_dataset(data_dir: Path) -> DatasetDict:
    dataset = load_from_disk(data_dir)
    missing = [split for split in SPLITS if split not in dataset]
    if missing:
        raise ValueError(f"{data_dir} is missing split(s): {', '.join(missing)}")
    return dataset


def load_datasets(data_dir: Path) -> tuple[DatasetDict, list[str]]:
    concept_dirs = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_dir() and (path / "dataset_dict.json").exists()
    )
    if not concept_dirs:
        raise ValueError(f"No concept datasets found under {data_dir}")

    datasets = []
    concepts = []
    for concept_dir in concept_dirs:
        dataset = load_concept_dataset(concept_dir)
        datasets.append(add_concept_column(dataset, concept_dir.name))
        concepts.append(concept_dir.name)

    merged = DatasetDict(
        {
            split: concatenate_datasets([dataset[split] for dataset in datasets])
            for split in SPLITS
        }
    )
    return merged, concepts


def get_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError("Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN in .env before uploading.")
    return token


def resolve_repo_id(api: HfApi, repo_id: str) -> str:
    if "/" in repo_id:
        return repo_id
    username = api.whoami()["name"]
    return f"{username}/{repo_id}"


def main() -> None:
    load_dotenv()
    args = parse_args()
    dataset, concepts = load_datasets(args.data_dir)

    token = get_token()
    api = HfApi(token=token)
    repo_id = resolve_repo_id(api, args.repo_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = Path(tmpdir)
        for split, filename in SPLITS.items():
            dataset[split].to_parquet(str(export_dir / filename))
        write_readme(export_dir, repo_id, args.data_dir, dataset, concepts)

        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(export_dir),
            commit_message="Upload dataset splits",
        )

    print(f"Uploaded {args.data_dir} to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
