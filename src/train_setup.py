from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from trl import GRPOTrainer
from typing import Literal
from dotenv import load_dotenv

from src.data_preprocessing.load_dataset import (
    load_standard_dataset,
)
from src.logging import (
    init_wandb_run,
    save_wandb_run_info,
    setup_huggingface_hub,
    setup_wandb,
)
from src.utils import is_main_process
from src.data_preprocessing.prompt_formatting import render_dataset_prompts


@dataclass(frozen=True)
class RunPaths:
    output_dir: Path
    events_log_path: Path
    final_model_dir: Path


@dataclass(frozen=False)
class TrainingDataSetup:
    train_dataset: Dataset
    eval_dataset: Dataset | None


def setup_run(cfg: DictConfig) -> tuple[bool, str, RunPaths]:
    load_dotenv()
    setup_huggingface_hub()
    wandb_enabled = setup_wandb()
    run_name = str(cfg.wandb.run_name)
    output_dir = Path(cfg.paths.storage_root) / "outputs" / run_name
    paths = RunPaths(
        output_dir=output_dir,
        events_log_path=output_dir / "events.jsonl",
        final_model_dir=output_dir / "final_model",
    )

    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        hydra_config_path = output_dir / "hydra_config.yaml"
        hydra_config_path.write_text(
            OmegaConf.to_yaml(cfg, resolve=True),
            encoding="utf-8",
        )
        if wandb_enabled:
            init_wandb_run(
                run_name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                config_yaml_path=hydra_config_path,
            )

    return wandb_enabled, run_name, paths


def setup_training_data(cfg: DictConfig, events_log_path: Path, training_mode: Literal["sft", "grpo"], tokenizer) -> TrainingDataSetup:

    train_dataset, test_dataset = load_standard_dataset(
        cfg.experiment.forget_concept,
        mode=training_mode,
        storage_root=cfg.paths.storage_root,
    )
    train_dataset = render_dataset_prompts(train_dataset, tokenizer)
    test_dataset = render_dataset_prompts(test_dataset, tokenizer)
    return TrainingDataSetup(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        rollout_func=None,
        prompt_buffer=None,
        data_generator=None,
    )


def finish_training(
    cfg: DictConfig,
    *,
    trainer: GRPOTrainer,
    tokenizer,
    model_name: str,
    forget_concept: str,
    paths: RunPaths,
    wandb_enabled: bool,
) -> None:
    if bool(cfg.training.grpo.get("save_final_model", True)):
        trainer.save_model(str(paths.final_model_dir))
        tokenizer.save_pretrained(paths.final_model_dir)

    if trainer.is_world_process_zero() and wandb_enabled:
        save_wandb_run_info(paths.output_dir)

    repo_name = (
        f"llm-unlearning-{model_name.split('/')[-1]}-forget-"
        f"{re.sub(r'[^A-Za-z0-9._-]+', '-', forget_concept).strip('-')}"
    )
    if cfg.experiment.push_to_hub and trainer.is_world_process_zero():
        trainer.model.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)
