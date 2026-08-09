from __future__ import annotations

from datasets import Dataset, load_from_disk
from typing import Literal

from src.data_preprocessing.utils import build_dataset_path, _is_main_process


def filter_empty_prompts(dataset: Dataset, prompt_column: str) -> Dataset:
    before = len(dataset)
    dataset = dataset.filter(
        lambda row: isinstance(row[prompt_column], str) and row[prompt_column].strip() != ""
    )
    removed = before - len(dataset)
    if removed > 0 and _is_main_process():
        print(
            f"WARNING: filtered out {removed} row(s) with empty prompts from "
            f"before applying dataset_size.",
            flush=True,
        )
    if len(dataset) == 0:
        raise ValueError(f"No valid prompts remain in dataset after filtering empty prompts.")
    return dataset


def load_standard_dataset(
    forget_concept: str,
    mode: Literal["sft", "grpo"],
    storage_root: str = ".",
) -> tuple[Dataset, Dataset]:

    dataset_path = build_dataset_path(forget_concept, storage_root=storage_root)
    dataset = load_from_disk(dataset_path)

    if mode in ["sft", "grpo"]:
        print(f"Reading data from {dataset_path}")
        train = dataset[f"{mode}/train"]
        test = dataset[f"{mode}/test"]
    else:
        raise ValueError(f"Mode must be 'sft' or 'grpo', not {mode}")

    return train, test
