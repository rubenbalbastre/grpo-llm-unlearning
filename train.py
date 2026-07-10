from __future__ import annotations

from importlib import import_module

import hydra
from omegaconf import DictConfig
from transformers import set_seed
from trl import GRPOConfig, GRPOTrainer

from src.peft import get_peft_config
from src.train_setup import (
    attach_adaptive_state,
    finish_training,
    load_model_and_tokenizer,
    setup_run,
    setup_training_data
)
from src.callbacks import get_training_callbacks


def build_reward_funcs(reward_config: DictConfig, forget_concept: str):
    reward_type = str(reward_config.get("type", "")).strip()
    if not reward_type:
        raise ValueError("reward.type must be configured.")

    module_name = f"src.reward.reward_{reward_type}"
    try:
        reward_module = import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name != module_name:
            raise
        raise ValueError(
            f"Unsupported reward.type={reward_type!r}; expected module {module_name}."
        ) from e

    return reward_module.build_reward_funcs(
        reward_config=reward_config,
        forget_concept=forget_concept,
    )


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.experiment.seed))
    wandb_enabled, run_name, paths = setup_run(cfg)

    model_name = cfg.model.name
    forget_concept = cfg.experiment.forget_concept
    model, tokenizer = load_model_and_tokenizer(model_name)
    peft_config = get_peft_config(cfg, num_hidden_layers=model.config.num_hidden_layers)
    if peft_config is None:
        print("Training mode: full fine-tuning", flush=True)
    else:
        print("Training mode: PEFT/LoRA", flush=True)
    data_setup = setup_training_data(cfg, paths.events_log_path, tokenizer)

    reward_funcs = build_reward_funcs(forget_concept=forget_concept, reward_config=cfg.reward)

    args = GRPOConfig(
        output_dir=str(paths.output_dir),
        run_name=run_name,
        # seeds
        seed=cfg.experiment.seed,
        data_seed=cfg.experiment.seed,
        # batch size
        per_device_train_batch_size=cfg.training.grpo.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.grpo.gradient_accumulation_steps,
        num_generations=cfg.training.grpo.num_generations,
        steps_per_generation=cfg.training.grpo.steps_per_generation,
        # sampling
        temperature=cfg.training.grpo.temperature,
        top_p=cfg.training.grpo.top_p,
        max_completion_length=cfg.training.grpo.max_completion_length,
        # logging
        remove_unused_columns=cfg.training.grpo.remove_unused_columns,
        report_to=["wandb"] if wandb_enabled else [],
        log_completions=cfg.training.grpo.log_completions,
        num_completions_to_print=0,
        logging_steps=cfg.training.grpo.logging_steps,
        # learning rate, scheduler
        learning_rate=cfg.training.grpo.learning_rate,
        lr_scheduler_type=cfg.training.grpo.lr_scheduler_type,
        lr_scheduler_kwargs={"min_lr_rate": 0.1} if cfg.training.grpo.lr_scheduler_type == "cosine_with_min_lr" else None,
        warmup_ratio=cfg.training.grpo.warmup_ratio,
        # eval
        eval_strategy=cfg.training.grpo.eval_strategy,
        eval_steps=cfg.training.grpo.eval_steps,
        per_device_eval_batch_size=cfg.training.grpo.per_device_eval_batch_size,
        num_generations_eval=cfg.training.grpo.num_generations_eval,
        # eval_on_start=True,
        # eval_accumulation_steps=1,
        # do_eval=True,
        # others
        num_iterations=cfg.training.grpo.num_iterations,
        beta=cfg.training.grpo.beta,
        gradient_checkpointing=cfg.training.grpo.gradient_checkpointing,
        fp16=cfg.training.grpo.fp16,
        bf16=cfg.training.grpo.bf16,
        num_train_epochs=cfg.training.grpo.num_train_epochs,
        max_steps=cfg.training.grpo.max_steps,
        ddp_find_unused_parameters=cfg.training.grpo.ddp_find_unused_parameters
    )

    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=data_setup.train_dataset,
        eval_dataset=data_setup.eval_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        rollout_func=data_setup.rollout_func,
        peft_config=peft_config,
        callbacks=get_training_callbacks(cfg.training.grpo.get("callback")),
    )
    trainer.events_log_path = paths.events_log_path
    attach_adaptive_state(trainer, data_setup)

    trainer.train()
    finish_training(
        cfg,
        trainer=trainer,
        tokenizer=tokenizer,
        model_name=model_name,
        forget_concept=forget_concept,
        paths=paths,
        wandb_enabled=wandb_enabled,
    )


if __name__ == "__main__":
    main()
