from __future__ import annotations

import os
import random
import re
import uuid
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from dotenv import load_dotenv

from src.data_generator import DataGenerator, PromptBuffer
from src.logging import setup_wandb, setup_huggingface_hub
from src.reward_function import make_unlearning_reward_func
from src.get_contaminated_data import get_rwku_contaminated_data


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
    dataset = load_dataset(
        cfg.standard_data.name,
        cfg.standard_data.config_name,
        split=cfg.standard_data.split,
    )
    subject_column = cfg.standard_data.subject_column
    prompt_column = cfg.standard_data.prompt_column
    dataset = dataset.filter(
        lambda row: row[subject_column] == cfg.experiment.forget_concept
    )
    return dataset.select_columns([prompt_column]).rename_column(prompt_column, "prompt")


def get_peft_config(cfg: DictConfig, num_hidden_layers: int) -> LoraConfig:
    from peft import LoraConfig
    lora_args = cfg.get("peft", None).get("lora", None)
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
        layers_to_transform=list(
            range(
                num_hidden_layers - lora_args.number_layers_to_transform,
                num_hidden_layers,
            )
        ),
    )


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:

    # seed for reproducibility
    random.seed(cfg.experiment.seed)
    torch.manual_seed(cfg.experiment.seed)

    # logs
    load_dotenv()
    setup_huggingface_hub()
    wandb_enabled = setup_wandb()
    output_dir = Path(cfg.paths.output_dir)
    events_log_path = Path(cfg.paths.events_log)
    final_model_dir = Path(cfg.paths.final_model_dir)

    # model name
    model_name = cfg.model.name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    peft_config = get_peft_config(cfg, num_hidden_layers=model.config.num_hidden_layers)

    local_batch_size = cfg.training.local_batch_size

    forget_concept = cfg.experiment.forget_concept
    mode = cfg.training.mode
    if mode == "adaptive":
        buffer = PromptBuffer(
            max_prompts=cfg.buffer.max_prompts,
            high_std_threshold=cfg.buffer.high_std_threshold,
            high_mean_threshold=cfg.buffer.high_mean_threshold,
        )
        num_processes = int(os.getenv("WORLD_SIZE", "1"))
        placeholder_dataset_size = local_batch_size * num_processes
        train_dataset = Dataset.from_dict(
            {"prompt": [""] * placeholder_dataset_size}
        )
        rollout_func = adaptive_rollout_func
        data_generator = DataGenerator(
            topic=f"Forget concept: '{forget_concept}.'",
            log_path=events_log_path,
            model_name=cfg.data_generator.model_name,
            history_size=cfg.data_generator.history_size,
            new_prompts_per_step=cfg.data_generator.new_prompts_per_step,
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
    else:
        raise ValueError("training.mode must be either 'adaptive' or 'standard'.")
    reward_func = make_unlearning_reward_func(buffer, events_log_path)

    args = GRPOConfig(
        output_dir=str(output_dir),
        run_name=f"{cfg.wandb.run_name_prefix}-{uuid.uuid4().hex[:8]}",
        per_device_train_batch_size=local_batch_size,
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
    )

    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        rollout_func=rollout_func,
        peft_config=peft_config,
    )
    trainer.events_log_path = events_log_path
    if data_generator is not None:
        trainer.prompt_buffer = buffer
        trainer.data_generator = data_generator

    # init training
    trainer.train()

    # save final model and tokenizer
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(final_model_dir)
    repo_name = (
        f"llm-unlearning-{model_name.split('/')[-1]}-forget-"
        f"{re.sub(r'[^A-Za-z0-9._-]+', '-', forget_concept).strip('-')}"
    )
    if cfg.experiment.push_to_hub and trainer.is_world_process_zero():
        trainer.model.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)


if __name__ == "__main__":
    main()
