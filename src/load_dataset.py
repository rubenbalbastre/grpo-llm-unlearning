from __future__ import annotations

import os

from datasets import Dataset, load_dataset
from omegaconf import DictConfig


def _is_main_process() -> bool:
    return int(os.getenv("RANK", "0")) == 0


def filter_empty_prompts(dataset: Dataset, prompt_column: str, dataset_label: str) -> Dataset:
    before = len(dataset)
    dataset = dataset.filter(
        lambda row: isinstance(row[prompt_column], str) and row[prompt_column].strip() != ""
    )
    removed = before - len(dataset)
    if removed > 0 and _is_main_process():
        print(
            f"WARNING: filtered out {removed} row(s) with empty prompts from "
            f"{dataset_label} before applying dataset_size.",
            flush=True,
        )
    if len(dataset) == 0:
        raise ValueError(f"No valid prompts remain in {dataset_label} after filtering empty prompts.")
    return dataset


def load_standard_dataset(cfg: DictConfig) -> Dataset:
    config_name = cfg.standard_data.get("config_name")
    dataset = load_dataset(
        cfg.standard_data.name,
        config_name if config_name else None,
        split=cfg.standard_data.split,
    )
    subject_column = cfg.standard_data.subject_column
    prompt_column = cfg.standard_data.prompt_column
    dataset = dataset.filter(
        lambda row: row[subject_column] == cfg.experiment.forget_concept
    )
    dataset = filter_empty_prompts(dataset, prompt_column, "standard_data")
    dataset_size = cfg.standard_data.get("dataset_size")
    if dataset_size is not None:
        dataset_size = int(dataset_size)
        if dataset_size <= 0:
            raise ValueError("standard_data.dataset_size must be positive or null.")
        dataset = dataset.shuffle(seed=cfg.standard_data.shuffle_seed)
        dataset = dataset.select(range(min(dataset_size, len(dataset))))
    dataset = dataset.select_columns([prompt_column, "output"])
    dataset = dataset.rename_column("output", "completion")
    if prompt_column != "prompt":
        dataset = dataset.rename_column(prompt_column, "prompt")

    # make dataset natural-question ONLY
    dataset = dataset.map(lambda example: {"prompt": example["prompt"].split("?")[0] + "?"})
    return dataset


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


def load_offline_dataset(cfg: DictConfig) -> Dataset:
    dataset = load_dataset(
        "json",
        data_files=str(cfg.offline_data.path),
        split=cfg.offline_data.split,
    )
    prompt_column = cfg.offline_data.prompt_column
    if prompt_column not in dataset.column_names:
        raise ValueError(
            f"Prompt column '{prompt_column}' not found in offline dataset columns: "
            f"{dataset.column_names}"
        )
    dataset = filter_empty_prompts(dataset, prompt_column, "offline_data")
    dataset_size = cfg.offline_data.get("dataset_size")
    if dataset_size is not None:
        dataset_size = int(dataset_size)
        if dataset_size <= 0:
            raise ValueError("offline_data.dataset_size must be positive or null.")
        dataset = dataset.select(range(min(dataset_size, len(dataset))))
    dataset = dataset.select_columns([prompt_column])
    if prompt_column != "prompt":
        dataset = dataset.rename_column(prompt_column, "prompt")
    return dataset
