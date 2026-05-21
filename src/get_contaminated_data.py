from datasets import load_dataset
from typing import Any, List


RWKU_FORGET_SPLITS = ("forget_level1", "forget_level2", "forget_level3")
RWKU_NEIGHBOR_SPLITS = ("neighbor_level1", "neighbor_level2")
RWKU_MIA_SPLITS = ("mia_forget", "mia_retain")


def get_tofu_contaminated_data() -> List[str]:
    pass


def get_rwku_contaminated_data(topic: str) -> List[str]:

    contaminated_data: list[str] = []

    for split_name in (*RWKU_FORGET_SPLITS, *RWKU_NEIGHBOR_SPLITS, *RWKU_MIA_SPLITS):
        print(f"Loading RWKU split '{split_name}' for topic '{topic}'...")
        dataset = dataset = load_dataset("jinzhuoran/RWKU", split_name, split="test")
        dataset = dataset.filter(lambda row: row["subject"] == topic)
        if "query" in dataset.column_names:
            contaminated_data.extend([example["query"] for example in dataset])
        elif "text" in dataset.column_names:
            contaminated_data.extend([example["text"] for example in dataset])

    return contaminated_data
