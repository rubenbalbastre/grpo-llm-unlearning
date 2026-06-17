from __future__ import annotations

import hydra
from omegaconf import DictConfig
from transformers import set_seed
from trl import GRPOConfig, GRPOTrainer

from src.peft import get_peft_config
from src.reward.factory import build_reward_funcs, get_reward_weights
from src.train_setup import (
    attach_adaptive_state,
    finish_training,
    load_model_and_tokenizer,
    setup_run,
    setup_training_data,
    standard_training_args,
)
from src.trainer_callback import get_training_callbacks


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.experiment.seed))
    wandb_enabled, run_name, paths = setup_run(cfg)

    model_name = cfg.model.name
    forget_concept = cfg.experiment.forget_concept
    model, tokenizer = load_model_and_tokenizer(model_name)
    peft_config = get_peft_config(cfg, num_hidden_layers=model.config.num_hidden_layers)
    data_setup = setup_training_data(cfg, paths.events_log_path)

    reward_funcs = build_reward_funcs(forget_concept=forget_concept, reward_config=cfg.reward)
    reward_weights = get_reward_weights(reward_config=cfg.reward)

    args = GRPOConfig(
        output_dir=str(paths.output_dir),
        run_name=run_name,
        seed=cfg.experiment.seed,
        data_seed=cfg.experiment.seed,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
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
        **standard_training_args(cfg),
    )

    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=data_setup.train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        rollout_func=data_setup.rollout_func,
        peft_config=peft_config,
        callbacks=get_training_callbacks(cfg.training.get("callback")),
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
