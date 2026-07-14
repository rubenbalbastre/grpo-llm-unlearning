import os


def _is_main_process() -> bool:
    return int(os.getenv("RANK", "0")) == 0


def build_dataset_path(target: str):
    return f"./data/{target}/"
