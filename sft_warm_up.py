import hydra
from omegaconf import DictConfig, open_dict
from trl import SFTTrainer, SFTConfig
from transformers import set_seed

from src.callbacks import SFTCallback
from src.peft import get_peft_config
from src.reward.components.avoid_refusal import DEFAULT_GARAK_REFUSAL_MODEL
from src.reward.reward_r2 import build_reward_funcs as build_r2_reward_funcs
from src.train_setup import load_model_and_tokenizer, setup_run, setup_training_data, finish_training


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:

    set_seed(int(cfg.experiment.seed))
    with open_dict(cfg):
        cfg.wandb.run_name = "-sft-".join(cfg.wandb.run_name.split("-"))
    wandb_enabled, run_name, paths = setup_run(cfg)

    model_name = cfg.model.name
    model, tokenizer = load_model_and_tokenizer(model_name)
    peft_config = get_peft_config(cfg, num_hidden_layers=model.config.num_hidden_layers)
    
    forget_concept = cfg.experiment.forget_concept
    dataset = setup_training_data(cfg, paths.events_log_path, training_mode="sft", tokenizer=tokenizer)
    # NOTE: custom change
    if cfg.training.sft.objective == "broad"
        print("SFT OBJECTIVE: BROAD. Selecting broad_completion as completion.")
        dataset.train_dataset = dataset.train_dataset.remove_columns("completion").rename_column("broad_completion", "completion")
        dataset.eval_dataset = dataset.eval_dataset.remove_columns("completion").rename_column("broad_completion", "completion")

    trainer_config = SFTConfig(
        output_dir=str(paths.output_dir),
        run_name=run_name,
        learning_rate=cfg.training.sft.learning_rate,
        lr_scheduler_type=cfg.training.sft.lr_scheduler_type,
        warmup_steps=cfg.training.sft.warmup_steps,
        bf16=cfg.training.sft.bf16,
        fp16=cfg.training.sft.fp16,
        gradient_checkpointing=cfg.training.sft.gradient_checkpointing,
        report_to="wandb" if wandb_enabled else None,
        eval_strategy=cfg.training.sft.eval_strategy,
        eval_steps=cfg.training.sft.eval_steps,
        logging_steps=cfg.training.sft.logging_steps,
        save_strategy=cfg.training.sft.save_strategy,
        save_steps=cfg.training.sft.save_steps,
        eval_on_start=True,
        per_device_train_batch_size=cfg.training.sft.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.sft.per_device_eval_batch_size,
        num_train_epochs=cfg.training.sft.num_train_epochs,
    )

    sft_monitoring_cfg = cfg.training.sft.get("monitoring", {})
    generation_monitoring_cfg = sft_monitoring_cfg.get("generation", {})
    classifier_monitoring_cfg = sft_monitoring_cfg.get("classifier", {})
    stop_monitoring_cfg = sft_monitoring_cfg.get("stop_conditions", {})
    r2_reward_func = build_r2_reward_funcs(
        reward_config=cfg.reward,
        forget_concept=forget_concept,
    )[0]
    sft_callback = SFTCallback(
        eval_dataset=dataset.eval_dataset,
        tokenizer=tokenizer,
        num_generations=generation_monitoring_cfg.get("num_generations", 8),
        max_prompts=generation_monitoring_cfg.get("max_prompts", 32),
        batch_size=generation_monitoring_cfg.get("batch_size", 4),
        max_new_tokens=generation_monitoring_cfg.get("max_new_tokens", 128),
        temperature=generation_monitoring_cfg.get("temperature", 1.0),
        top_p=generation_monitoring_cfg.get("top_p", 1.0),
        log_completions=generation_monitoring_cfg.get("log_completions", True),
        stop_refusal_completion_rate_threshold=stop_monitoring_cfg.get(
            "refusal_completion_rate",
            0.10,
        ),
        stop_r2_reward_threshold=stop_monitoring_cfg.get(
            "r2_reward_threshold",
            0.05,
        ),
        classifier_model_name=classifier_monitoring_cfg.get(
            "model_name", DEFAULT_GARAK_REFUSAL_MODEL
        ),
        classifier_batch_size=classifier_monitoring_cfg.get("batch_size", 32),
        classifier_threshold=classifier_monitoring_cfg.get("threshold", 0.5),
        r2_reward_func=r2_reward_func,
    )

    trainer = SFTTrainer(
        args=trainer_config,
        peft_config=peft_config,
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset.train_dataset,
        eval_dataset=dataset.eval_dataset,
        callbacks=[sft_callback],
    )
    sft_callback.bind_trainer(trainer)

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
