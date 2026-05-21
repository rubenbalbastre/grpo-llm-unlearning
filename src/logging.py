import os
import json
import openai
from pathlib import Path
from typing import Any, List
import wandb
from huggingface_hub import login


def log_event(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def setup_wandb() -> bool:
    """Auto-login to Weights & Biases from environment if available."""
    import wandb

    api_key = os.getenv("WANDB_API_KEY", "").strip()
    if api_key:
        wandb.login(key=api_key, relogin=True)
    else:
        try:
            wandb.login()
        except Exception:
            return False
    return True


def setup_huggingface_hub() -> None:
    login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))