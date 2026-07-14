from __future__ import annotations

import os

from datasets import Dataset, load_dataset, load_from_disk
from omegaconf import DictConfig
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


def load_standard_dataset(cfg: DictConfig, mode: Literal["sft", "grpo"]) -> Dataset:

    dataset_path = build_dataset_path(cfg.experiment.forget_concept)
    dataset = load_from_disk(dataset_path)

    if mode in ["sft", "grpo"]:
        print(f"Reading data from {dataset_path}")
        train = dataset[f"{mode}/train"]
        test = dataset[f"{mode}/test"]
    else:
        raise ValueError(f"Mode must be 'sft' or 'grpo', not {mode}")

    return train, test


def load_initial_adaptive_prompts(cfg: DictConfig) -> list[str]:
    prompt_count = cfg.data_generator.get("initial_standard_prompt_count")
    if prompt_count is None:
        return []
    prompt_count = int(prompt_count)
    if prompt_count < 0:
        raise ValueError("data_generator.initial_standard_prompt_count must be non-negative or null.")
    if prompt_count == 0:
        return []
    if prompt_count > int(cfg.buffer.max_prompts):
        raise ValueError(
            "data_generator.initial_standard_prompt_count cannot exceed "
            f"buffer.max_prompts: requested {prompt_count}, "
            f"buffer.max_prompts={cfg.buffer.max_prompts}."
        )

    dataset = load_standard_dataset(cfg)
    if prompt_count > len(dataset):
        raise ValueError(
            "data_generator.initial_standard_prompt_count cannot exceed the "
            f"standard dataset size after filtering: requested {prompt_count}, "
            f"available {len(dataset)}."
        )
    dataset = dataset.shuffle(seed=int(cfg.experiment.seed))
    dataset = dataset.select(range(prompt_count))
    return [str(prompt).strip() for prompt in dataset["prompt"]]
