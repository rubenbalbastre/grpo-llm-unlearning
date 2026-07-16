from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from trl import GRPOTrainer
from typing import Literal
from dotenv import load_dotenv

from src.data_generator.data_generator import DataGenerator
from src.data_generator.get_contaminated_data import get_rwku_contaminated_data
from src.data_generator.prompt_buffer import PromptBuffer
from src.data_preprocessing.load_dataset import (
    load_initial_adaptive_prompts,
    load_standard_dataset,
)
from src.logging import (
    init_wandb_run,
    save_wandb_run_info,
    setup_huggingface_hub,
    setup_wandb,
)
from src.model_loading import load_model_and_tokenizer
from src.utils import is_main_process
from src.data_preprocessing.prompt_formatting import render_chat_prompt, render_dataset_prompts


@dataclass(frozen=True)
class RunPaths:
    output_dir: Path
    events_log_path: Path
    final_model_dir: Path


@dataclass(frozen=False)
class TrainingDataSetup:
    train_dataset: Dataset
    eval_dataset: Dataset | None
    rollout_func: Callable | None
    prompt_buffer: PromptBuffer | None
    data_generator: DataGenerator | None


def adaptive_rollout_func(prompts: list[str], trainer) -> dict[str, Any]:
    step = int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0)
    batch_size = len(prompts)
    if batch_size == 0:
        raise RuntimeError("adaptive_rollout_func received an empty prompt batch.")
    num_processes = int(getattr(trainer.accelerator, "num_processes", 1))
    process_index = int(getattr(trainer.accelerator, "process_index", 0))
    global_batch_size = batch_size * num_processes
    if global_batch_size % trainer.num_generations != 0:
        raise RuntimeError(
            f"Global rollout batch size {global_batch_size} must be divisible by "
            f"num_generations={trainer.num_generations}."
        )
    num_prompts = global_batch_size // trainer.num_generations

    selected = trainer.data_generator.generate(
        buffer=trainer.prompt_buffer,
        num_prompts=num_prompts,
        step=step,
        accelerator=trainer.accelerator,
    )
    expanded = [prompt for prompt in selected for _ in range(trainer.num_generations)]
    start = process_index * batch_size
    final = expanded[start : start + batch_size]
    if len(final) != batch_size:
        raise RuntimeError(f"Adaptive rollout selected {len(final)} prompts, expected {batch_size}.")

    rendered = [render_chat_prompt(trainer.processing_class, prompt) for prompt in final]
    prompt_ids, images, multimodal_fields = trainer._tokenize_prompts(rendered)
    completion_ids, logprobs = trainer._generate_single_turn(prompt_ids, images, multimodal_fields)

    if len(completion_ids) != len(prompt_ids):
        raise RuntimeError(
            f"Unexpected rollout shapes: len(prompt_ids)={len(prompt_ids)}, "
            f"len(completion_ids)={len(completion_ids)}."
        )

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "selected_prompt": final,
    }


def setup_run(cfg: DictConfig) -> tuple[bool, str, RunPaths]:
    load_dotenv()
    setup_huggingface_hub()
    wandb_enabled = setup_wandb()
    run_name = str(cfg.wandb.run_name)
    output_dir = Path(cfg.paths.output_root) / run_name
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


def build_prompt_buffer(cfg: DictConfig) -> PromptBuffer:
    return PromptBuffer(
        max_prompts=cfg.buffer.max_prompts,
        high_reward_std_threshold=cfg.buffer.high_reward_std_threshold,
        high_mean_reward_threshold=cfg.buffer.high_mean_reward_threshold,
    )


def seed_adaptive_buffer(buffer: PromptBuffer, cfg: DictConfig) -> None:
    initial_prompts = load_initial_adaptive_prompts(cfg)
    buffer.add_generated_prompts(initial_prompts)
    if initial_prompts and is_main_process():
        print(
            f"Seeded adaptive prompt buffer with {len(initial_prompts)} "
            "standard_data prompt(s).",
            flush=True,
        )


def adaptive_placeholder_dataset(cfg: DictConfig) -> Dataset:
    num_processes = int(os.getenv("WORLD_SIZE", "1"))
    placeholder_dataset_size = (
        cfg.training.grpo.per_device_train_batch_size
        * num_processes
        * cfg.training.grpo.steps_per_generation
        // cfg.training.grpo.num_generations
    )
    return Dataset.from_dict({"prompt": [""] * placeholder_dataset_size})


def build_data_generator(cfg: DictConfig, events_log_path: Path) -> DataGenerator:
    forget_concept = cfg.experiment.forget_concept
    data_proposer_cfg = cfg.data_generator.get("data_proposer")
    if data_proposer_cfg is None:
        data_proposer_model_name = cfg.data_generator.model_name
        data_proposer_mode = cfg.data_generator.mode
        objective = cfg.data_generator.get("objective", "general")
        proposer_context_cfg = {}
        proposer_execution_cfg = {}
    else:
        data_proposer_model_name = data_proposer_cfg.model_name
        data_proposer_mode = data_proposer_cfg.mode
        objective = data_proposer_cfg.get("objective", "general")
        proposer_context_cfg = data_proposer_cfg.get("proposer_context", {})
        proposer_execution_cfg = data_proposer_cfg.get("execution", {})

    return DataGenerator(
        topic=f"Forget concept: '{forget_concept}.'",
        log_path=events_log_path,
        model_name=data_proposer_model_name,
        objective=objective,
        mode=data_proposer_mode,
        context_mode=proposer_context_cfg.get("context_mode", "summary"),
        context_max_prompts=proposer_context_cfg.get(
            "context_max_prompts",
            cfg.data_generator.get("generation_context_size", 16),
        ),
        context_max_completions_per_prompt=proposer_context_cfg.get(
            "context_max_completions_per_prompt",
            2,
        ),
        execution_mode=proposer_execution_cfg.get("mode", "batch"),
        max_concurrent_requests=proposer_execution_cfg.get("max_concurrent_requests", 4),
        prompts_per_context_item=proposer_execution_cfg.get("prompts_per_context_item", 2),
        new_prompts_per_step=cfg.data_generator.new_prompts_per_step,
        max_generated_prompts=cfg.data_generator.get("max_generated_prompts"),
        safe=cfg.data_generator.safe,
        protected_data=(
            get_rwku_contaminated_data(forget_concept)
            if cfg.data_generator.safe
            else None
        ),
    )


def setup_training_data(cfg: DictConfig, events_log_path: Path, training_mode: Literal["sft", "grpo"], tokenizer) -> TrainingDataSetup:
    data_mode = cfg.training.grpo.mode
    if data_mode == "adaptive":
        buffer = build_prompt_buffer(cfg)
        seed_adaptive_buffer(buffer, cfg)
        return TrainingDataSetup(
            train_dataset=adaptive_placeholder_dataset(cfg),
            rollout_func=adaptive_rollout_func,
            prompt_buffer=buffer,
            data_generator=build_data_generator(cfg, events_log_path),
        )
    elif data_mode == "standard":
        train_dataset, test_dataset = load_standard_dataset(cfg, mode=training_mode)
        train_dataset = render_dataset_prompts(train_dataset, tokenizer)
        test_dataset = render_dataset_prompts(test_dataset, tokenizer)
        return TrainingDataSetup(
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            rollout_func=None,
            prompt_buffer=None,
            data_generator=None,
        )
    else:
        raise ValueError("training.grpo.mode must be one of: 'adaptive' or 'standard'.")


def attach_adaptive_state(trainer: GRPOTrainer, data_setup: TrainingDataSetup) -> None:
    if data_setup.data_generator is None:
        return
    trainer.prompt_buffer = data_setup.prompt_buffer
    trainer.data_generator = data_setup.data_generator


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
