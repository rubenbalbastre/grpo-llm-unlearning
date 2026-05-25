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

from src.data_generator import AdaptivePromptBuffer, DataGenerator
from src.logging import setup_wandb, setup_huggingface_hub
from src.reward_function import make_unlearning_reward_func
from src.get_contaminated_data import get_rwku_contaminated_data


def adaptive_rollout_func(prompts: list[str], trainer) -> dict[str, Any]:

    step = int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0)
    batch_size = len(prompts)
    if batch_size == 0:
        raise RuntimeError("adaptive_rollout_func received an empty prompt batch.")

    final = trainer.data_generator.generate(
        buffer=trainer.prompt_buffer,
        batch_size=batch_size,
        step=step,
        accelerator=trainer.accelerator,
    )

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
    }


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
    peft_config = get_peft_config(cfg, num_hidden_layers=model.config.num_hidden_layers)

    num_processes = int(os.getenv("WORLD_SIZE", "1"))
    local_batch_size = cfg.training.local_batch_size
    G = cfg.training.num_generations
    placeholder_dataset_size = local_batch_size * num_processes
    placeholder_dataset = Dataset.from_dict(
        {
            "prompt": [""] * placeholder_dataset_size,
        }
    )

    # adaptive GRPO setup
    buffer = AdaptivePromptBuffer(max_history=cfg.buffer.max_history)
    reward_func = make_unlearning_reward_func(buffer, events_log_path)
    forget_concept = cfg.experiment.forget_concept
    data_generator = DataGenerator(
        topic=f"Forget concept: '{forget_concept}.'",
        log_path=events_log_path,
        model_name=cfg.data_generator.model_name,
        history_size=cfg.data_generator.history_size,
        safe=cfg.data_generator.safe,
        protected_data=get_rwku_contaminated_data(forget_concept) if cfg.data_generator.safe else None,
    )

    args = GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=local_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_generations=G,
        steps_per_generation=cfg.training.steps_per_generation,
        num_iterations=cfg.training.num_iterations,
        beta=cfg.training.beta,
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
        train_dataset=placeholder_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        rollout_func=adaptive_rollout_func,
        peft_config=peft_config,
    )
    trainer.prompt_buffer = buffer
    trainer.data_generator = data_generator
    trainer.events_log_path = events_log_path

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
