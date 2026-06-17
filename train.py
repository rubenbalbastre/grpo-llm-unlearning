from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer
from dotenv import load_dotenv

from src.data_generator.data_generator import DataGenerator
from src.logging import (
    init_wandb_run,
    save_wandb_run_info,
    setup_wandb,
    setup_huggingface_hub,
)
from src.data_generator.prompt_buffer import PromptBuffer
from src.reward.factory import build_reward_functions, get_reward_weights
from src.data_generator.get_contaminated_data import get_rwku_contaminated_data
from src.trainer_callback import get_training_callbacks


def is_main_process() -> bool:
    return int(os.getenv("RANK", "0")) == 0


def filter_empty_prompts(dataset: Dataset, prompt_column: str, dataset_label: str) -> Dataset:
    before = len(dataset)
    dataset = dataset.filter(
        lambda row: isinstance(row[prompt_column], str) and row[prompt_column].strip() != ""
    )
    removed = before - len(dataset)
    if removed > 0 and is_main_process():
        print(
            f"WARNING: filtered out {removed} row(s) with empty prompts from "
            f"{dataset_label} before applying dataset_size.",
            flush=True,
        )
    if len(dataset) == 0:
        raise ValueError(f"No valid prompts remain in {dataset_label} after filtering empty prompts.")
    return dataset


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

    # tokenize and generate completions for the final prompts
    prompt_ids, images, multimodal_fields = trainer._tokenize_prompts(final)
    completion_ids, logprobs = trainer._generate_single_turn(prompt_ids, images, multimodal_fields)

    n_completions = len(completion_ids)
    if n_completions != len(prompt_ids):
        raise RuntimeError(
            f"Unexpected rollout shapes: len(prompt_ids)={len(prompt_ids)}, "
            f"len(completion_ids)={n_completions}."
        )

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "selected_prompt": final,
    }


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
        dataset = dataset.select(range(min(dataset_size, len(dataset))))
    dataset = dataset.select_columns([prompt_column])
    if prompt_column != "prompt":
        dataset = dataset.rename_column(prompt_column, "prompt")
    return dataset


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


def get_peft_config(cfg: DictConfig, num_hidden_layers: int) -> LoraConfig:
    from peft import LoraConfig
    lora_args = cfg.get("peft", None).get("lora", None)
    number_layers_to_transform = int(lora_args.number_layers_to_transform)
    if number_layers_to_transform == -1:
        layers_to_transform = list(range(num_hidden_layers))
    elif 1 <= number_layers_to_transform <= num_hidden_layers:
        layers_to_transform = list(
            range(num_hidden_layers - number_layers_to_transform, num_hidden_layers)
        )
    else:
        raise ValueError(
            "peft.lora.number_layers_to_transform must be -1 or between 1 "
            f"and {num_hidden_layers}, got {number_layers_to_transform}."
        )
    return LoraConfig(
        r=lora_args.r,
        lora_alpha=lora_args.alpha,
        init_lora_weights=lora_args.init_lora_weights,
        target_modules=OmegaConf.to_container(
            lora_args.target_modules, resolve=True
        ),
        task_type="CAUSAL_LM",
        bias="none",
        layers_pattern="layers",
        layers_to_transform=layers_to_transform,
    )


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:

    # seed for reproducibility
    set_seed(int(cfg.experiment.seed))

    # logs
    load_dotenv()
    setup_huggingface_hub()
    wandb_enabled = setup_wandb()
    run_name = str(cfg.wandb.run_name)
    output_dir = Path(cfg.paths.output_root) / run_name
    events_log_path = output_dir / "events.jsonl"
    final_model_dir = output_dir / "final_model"
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    hydra_config_path = output_dir / "hydra_config.yaml"
    if int(os.getenv("RANK", "0")) == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        hydra_config_path.write_text(
            OmegaConf.to_yaml(cfg, resolve=True),
            encoding="utf-8",
        )
        if wandb_enabled:
            init_wandb_run(
                run_name=run_name,
                config=resolved_cfg,
                config_yaml_path=hydra_config_path,
            )

    # model name
    model_name = cfg.model.name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # LoRA configuration
    peft_config = get_peft_config(cfg, num_hidden_layers=model.config.num_hidden_layers)

    # Data Generator configuration
    per_device_train_batch_size = cfg.training.per_device_train_batch_size
    forget_concept = cfg.experiment.forget_concept
    mode = cfg.training.mode
    if mode == "adaptive":
        data_proposer_cfg = cfg.data_generator.get("data_proposer")
        if data_proposer_cfg is None:
            data_proposer_model_name = cfg.data_generator.model_name
            data_proposer_mode = cfg.data_generator.mode
            proposer_context_cfg = {}
            proposer_execution_cfg = {}
        else:
            data_proposer_model_name = data_proposer_cfg.model_name
            data_proposer_mode = data_proposer_cfg.mode
            proposer_context_cfg = data_proposer_cfg.get("proposer_context", {})
            proposer_execution_cfg = data_proposer_cfg.get("execution", {})
        context_mode = proposer_context_cfg.get("context_mode", "summary")
        context_max_prompts = proposer_context_cfg.get(
            "context_max_prompts",
            cfg.data_generator.get("generation_context_size", 16),
        )
        context_max_completions_per_prompt = proposer_context_cfg.get(
            "context_max_completions_per_prompt",
            2,
        )
        execution_mode = proposer_execution_cfg.get("mode", "batch")
        max_concurrent_requests = proposer_execution_cfg.get("max_concurrent_requests", 4)
        prompts_per_context_item = proposer_execution_cfg.get("prompts_per_context_item", 2)
        buffer = PromptBuffer(
            max_prompts=cfg.buffer.max_prompts,
            high_reward_std_threshold=cfg.buffer.high_reward_std_threshold,
            high_mean_reward_threshold=cfg.buffer.high_mean_reward_threshold,
        )
        num_processes = int(os.getenv("WORLD_SIZE", "1"))
        placeholder_dataset_size = (
            per_device_train_batch_size
            * num_processes
            * cfg.training.steps_per_generation
            // cfg.training.num_generations
        )
        train_dataset = Dataset.from_dict(
            {"prompt": [""] * placeholder_dataset_size}
        )
        rollout_func = adaptive_rollout_func
        data_generator = DataGenerator(
            topic=f"Forget concept: '{forget_concept}.'",
            log_path=events_log_path,
            model_name=data_proposer_model_name,
            mode=data_proposer_mode,
            context_mode=context_mode,
            context_max_prompts=context_max_prompts,
            context_max_completions_per_prompt=context_max_completions_per_prompt,
            execution_mode=execution_mode,
            max_concurrent_requests=max_concurrent_requests,
            prompts_per_context_item=prompts_per_context_item,
            new_prompts_per_step=cfg.data_generator.new_prompts_per_step,
            max_generated_prompts=cfg.data_generator.get("max_generated_prompts"),
            safe=cfg.data_generator.safe,
            protected_data=(
                get_rwku_contaminated_data(forget_concept)
                if cfg.data_generator.safe
                else None
            ),
        )
    elif mode == "standard":
        buffer = None
        train_dataset = load_standard_dataset(cfg)
        rollout_func = None
        data_generator = None
    elif mode == "offline":
        buffer = None
        train_dataset = load_offline_dataset(cfg)
        rollout_func = None
        data_generator = None
    else:
        raise ValueError("training.mode must be one of: 'adaptive', 'standard', or 'offline'.")

    # Reward function configuration
    reward_funcs = build_reward_functions(cfg.reward, forget_concept)
    reward_weights = get_reward_weights(cfg.reward)

    # other GRPO configurations
    standard_training_args: dict[str, Any] = {}
    if mode in {"standard", "offline"}:
        num_epochs = float(cfg.training.num_epochs)
        if num_epochs <= 0:
            raise ValueError("training.num_epochs must be positive in standard/offline mode.")
        standard_training_args["num_train_epochs"] = num_epochs

    callbacks = get_training_callbacks(cfg.training.get("callback"))

    args = GRPOConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        seed=cfg.experiment.seed,
        data_seed=cfg.experiment.seed,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_generations=cfg.training.num_generations,
        steps_per_generation=cfg.training.steps_per_generation,
        num_iterations=cfg.training.num_iterations,
        beta=cfg.training.beta,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        max_steps=cfg.training.max_steps,
        learning_rate=cfg.training.learning_rate,
        max_completion_length=cfg.training.max_completion_length,
        remove_unused_columns=cfg.training.remove_unused_columns,
        report_to=["wandb"] if wandb_enabled else [],
        log_completions=cfg.training.log_completions,
        logging_steps=cfg.training.logging_steps,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        reward_weights=reward_weights,
        ddp_find_unused_parameters=cfg.training.ddp_find_unused_parameters,
        **standard_training_args,
    )

    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        rollout_func=rollout_func,
        peft_config=peft_config,
        callbacks=callbacks
    )
    trainer.events_log_path = events_log_path
    if data_generator is not None:
        trainer.prompt_buffer = buffer
        trainer.data_generator = data_generator

    # init training
    trainer.train()

    save_final_model = bool(cfg.training.get("save_final_model", True))
    if save_final_model:
        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(final_model_dir)
    if trainer.is_world_process_zero():
        if wandb_enabled:
            save_wandb_run_info(output_dir)
    repo_name = (
        f"llm-unlearning-{model_name.split('/')[-1]}-forget-"
        f"{re.sub(r'[^A-Za-z0-9._-]+', '-', forget_concept).strip('-')}"
    )
    if cfg.experiment.push_to_hub and trainer.is_world_process_zero():
        trainer.model.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)


if __name__ == "__main__":
    main()
