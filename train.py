from __future__ import annotations

import hydra
from omegaconf import DictConfig
from transformers import set_seed
from trl import GRPOTrainer

from src.peft import get_peft_config
from src.reward.factory import build_reward_funcs, get_reward_weights
from src.train_setup import (
    attach_adaptive_state,
    build_grpo_config,
    finish_training,
    load_model_and_tokenizer,
    setup_run,
    setup_training_data,
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
    args = build_grpo_config(
        cfg,
        output_dir=paths.output_dir,
        run_name=run_name,
        wandb_enabled=wandb_enabled,
        reward_weights=reward_weights,
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
