#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi


CONFIGS = {
    "grpo": {
        "train": "grpo/train",
        "test": "grpo/test",
    },
    "sft": {
        "train": "sft/train",
        "validation": "sft/test",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", nargs="?", default="grpo-unlearning-data", help="Hugging Face dataset repo.")
    parser.add_argument("output_dir", nargs="?", type=Path, default=Path("data"), help="Output data directory.")
    return parser.parse_args()


def get_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Set HF_TOKEN in .env before downloading.")
    return token


def resolve_repo_id(api: HfApi, repo_id: str) -> str:
    if "/" in repo_id:
        return repo_id
    username = api.whoami()["name"]
    return f"{username}/{repo_id}"


def without_concept_column(dataset: Dataset) -> Dataset:
    return dataset.remove_columns(["concept"]) if "concept" in dataset.column_names else dataset


def main() -> None:
    load_dotenv()
    args = parse_args()

    token = get_token()
    api = HfApi(token=token)
    repo_id = resolve_repo_id(api, args.repo_id)
    datasets = {
        name: load_dataset(repo_id, name=name, token=token)
        for name in CONFIGS
    }

    concepts = sorted(set(datasets["grpo"]["train"]["concept"]))
    for concept in concepts:
        splits = {}
        for config_name, config_splits in CONFIGS.items():
            for hub_split, local_split in config_splits.items():
                split_dataset = datasets[config_name][hub_split].filter(
                    lambda row: row["concept"] == concept
                )
                splits[local_split] = without_concept_column(split_dataset)

        output_path = args.output_dir / concept
        DatasetDict(splits).save_to_disk(output_path)
        print(f"Wrote {output_path}")

    print(f"Downloaded {repo_id} to {args.output_dir}")


if __name__ == "__main__":
    main()
