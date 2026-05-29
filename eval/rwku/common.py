from typing import List, Optional

from datasets import load_dataset


def load_rwku_split(config_name: str):
    loaded = load_dataset("jinzhuoran/RWKU", config_name)
    if "test" in loaded:
        return loaded["test"]
    if "train" in loaded:
        return loaded["train"]
    return loaded[next(iter(loaded.keys()))]


def filter_subjects(dataset, subjects: Optional[List[str]]):
    if not subjects or "subject" not in dataset.column_names:
        return dataset
    subject_set = set(subjects)
    return dataset.filter(lambda x: x["subject"] in subject_set)


def select_max_examples(dataset, max_examples: Optional[int]):
    if max_examples is None:
        return dataset
    return dataset.select(range(min(max_examples, len(dataset))))


def shard_dataset(dataset, shard) -> object:
    if shard is None:
        return dataset
    shard_count = int(shard.count)
    shard_index = int(shard.index)
    if shard_count == 1:
        return dataset
    return dataset.shard(
        num_shards=shard_count,
        index=shard_index,
        contiguous=True,
    )


def normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(normalize_text(v) for v in value if normalize_text(v)).strip()
    if isinstance(value, dict):
        for key in ("text", "answer", "content", "value"):
            if key in value:
                return normalize_text(value[key])
        return " ".join(normalize_text(v) for v in value.values() if normalize_text(v)).strip()
    return str(value).strip()


def first_present(example: dict, keys: List[str]):
    for key in keys:
        if key in example and example[key] is not None:
            return example[key]
    return None


def extract_text_for_likelihood(example: dict) -> str:
    return normalize_text(
        first_present(
            example,
            ["text", "passage", "document", "content", "input", "prompt", "query", "question", "answer"],
        )
    )
