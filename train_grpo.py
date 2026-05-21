from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any

import hydra
import torch
from datasets import Dataset
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from dotenv import load_dotenv

from src.logging import setup_wandb, setup_huggingface_hub
from src.reward_function import make_unlearning_reward_func


@hydra.main(version_base=None, config_path="configs", config_name="train_adaptive_grpo")
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

    num_processes = int(os.getenv("WORLD_SIZE", "1"))
    local_batch_size = cfg.training.local_batch_size
    G = cfg.training.num_generations

    # dataset
    placeholder_dataset_size = local_batch_size * num_processes
    forget_concept = cfg.experiment.forget_concept
    train_dataset = None # TODO: select dataset based on forget topic
    eval_dataset = None # TODO: select dataset based on benchmark

    # GRPO setup
    reward_func = make_unlearning_reward_func(buffer, events_log_path)

    args = GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=local_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_generations=G,
        steps_per_generation=cfg.training.steps_per_generation,
        num_iterations=cfg.training.num_iterations,
        # num_train_epochs=0.1,
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
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        rollout_func=adaptive_rollout_func,
    )

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
        model.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)


if __name__ == "__main__":
    main()
