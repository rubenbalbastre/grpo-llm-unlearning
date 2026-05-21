from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from dotenv import load_dotenv

from src.data_generator import AdaptivePromptBuffer, DataGenerator, DataGeneratorPool
from src.logging import setup_wandb, log_event, setup_huggingface_hub
from src.reward_function import make_unlearning_reward_func


def adaptive_rollout_func(prompts: list[str], trainer) -> dict[str, Any]:

    step = int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0)
    batch_size = len(prompts)
    if batch_size == 0:
        raise RuntimeError("adaptive_rollout_func received an empty prompt batch.")

    # select history for data generator based on current buffer
    history = trainer.prompt_buffer.select_history(k=16)
    trainer.data_generator_pool.maybe_refresh(
        step=step,
        history=history,
        log_path=trainer.events_log_path,
        accelerator=trainer.accelerator,
    )
    process_index = int(getattr(trainer.accelerator, "process_index", 0))
    selected = trainer.data_generator_pool.prompts_for_process(
        process_index=process_index,
        batch_size=batch_size,
    )
    final = [str(candidate["prompt"]).strip() for candidate in selected]

    # tokenize and generate completions for the final prompts
    prompt_ids, images, multimodal_fields = trainer._tokenize_prompts(final)
    completion_ids, logprobs = trainer._generate_single_turn(prompt_ids, images, multimodal_fields)

    n_completions = len(completion_ids)
    if n_completions != len(prompt_ids):
        raise RuntimeError(
            f"Unexpected rollout shapes: len(prompt_ids)={len(prompt_ids)}, "
            f"len(completion_ids)={n_completions}."
        )

    # logging
    out = {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
    }

    log_event(
        trainer.events_log_path,
        {
            "type": "rollout",
            "step": step,
            "batch_size": len(final),
            "history_size": len(history),
            "pool_size": len(trainer.data_generator_pool.candidates),
            "process_index": process_index,
            "final_prompts": final,
        },
    )
    return out


def main() -> None:

    # seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # logs
    load_dotenv()
    setup_huggingface_hub()
    wandb_enabled = setup_wandb()
    output_dir = Path("./outputs/adaptive_grpo_min")
    events_log_path = output_dir / "events.jsonl"
    final_model_dir = output_dir / "final_model"

    # model name
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name)

    num_processes = int(os.getenv("WORLD_SIZE", "1"))
    local_batch_size = 4
    G = 2
    data_generator_num_candidates = local_batch_size * num_processes
    placeholder_dataset = Dataset.from_dict(
        {
            "prompt": [""] * data_generator_num_candidates,
        }
    )

    # adaptive GRPO setup
    buffer = AdaptivePromptBuffer(max_history=512)
    forget_concept = "Stephen King"
    data_generator = DataGenerator(topic=f"Forget concept: '{forget_concept}.'")
    data_generator_refresh_every = 1
    data_generator_pool = DataGeneratorPool(
        generator=data_generator,
        num_candidates=data_generator_num_candidates,
        refresh_every=data_generator_refresh_every,
    )
    reward_func = make_unlearning_reward_func(buffer, events_log_path)

    args = GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=local_batch_size,
        gradient_accumulation_steps=1,
        num_generations=G,
        steps_per_generation=1,
        num_iterations=3,
        # num_train_epochs=0.1,
        max_steps=10,
        learning_rate=5e-6,
        max_completion_length=64,
        remove_unused_columns=False,
        report_to=["wandb"] if wandb_enabled else [],
        log_completions=True,
        logging_steps=1,
    )

    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=placeholder_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        rollout_func=adaptive_rollout_func,
    )
    trainer.prompt_buffer = buffer
    trainer.data_generator_pool = data_generator_pool
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
    if trainer.is_world_process_zero():
        model.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)


if __name__ == "__main__":
    main()
