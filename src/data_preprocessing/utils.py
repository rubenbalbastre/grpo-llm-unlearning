from pathlib import Path

from src.utils import is_main_process


def _is_main_process() -> bool:
    return is_main_process()


def build_dataset_path(target: str, storage_root: str | Path = ".") -> Path:
    return Path(storage_root) / "data" / target
