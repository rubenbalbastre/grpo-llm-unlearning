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


def init_wandb_run(
    run_name: str,
    config: dict[str, Any],
    config_yaml_path: Path | None = None,
) -> None:
    """Initialize the active training run before Trainer attaches W&B logging."""
    if wandb.run is not None:
        wandb.config.update({"hydra": config}, allow_val_change=True)
    else:
        wandb.init(name=run_name, config={"hydra": config})
    if config_yaml_path is not None:
        wandb.save(str(config_yaml_path), policy="now")


def setup_huggingface_hub() -> None:
    if int(os.getenv("RANK", "0")) == 0:
        login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))


def save_wandb_run_info(model_dir: Path) -> None:
    """Persist the active training run identity for post-training evaluation."""
    if wandb.run is None:
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    run_info = {
        "id": wandb.run.id,
        "name": wandb.run.name,
        "project": wandb.run.project,
        "entity": wandb.run.entity,
    }
    (model_dir / "wandb_run.json").write_text(
        json.dumps(run_info, indent=2),
        encoding="utf-8",
    )
