import hydra
from omegaconf import DictConfig, open_dict
from trl import SFTTrainer, SFTConfig
from transformers import set_seed

from src.callbacks import RefusalGenerationMetricCallback
from src.peft import get_peft_config
from src.reward.components.avoid_refusal import DEFAULT_GARAK_REFUSAL_MODEL
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
    dataset = setup_training_data(cfg, paths.events_log_path, tokenizer)
    sft_dataset = dataset.train_dataset.train_test_split(test_size=cfg.standard_data.sft_test_size + cfg.standard_data.sft_train_size, seed=cfg.standard_data.shuffle_seed, shuffle=True)["test"]
    sft_dataset_splits = sft_dataset.train_test_split(test_size=cfg.standard_data.sft_test_size / (cfg.standard_data.sft_test_size + cfg.standard_data.sft_train_size), seed=cfg.standard_data.shuffle_seed, shuffle=True)
    print(sft_dataset_splits["train"].num_rows)
    print(sft_dataset_splits["test"].num_rows)

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
        eval_on_start=True,
        per_device_train_batch_size=cfg.training.sft.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.sft.per_device_eval_batch_size,
        num_train_epochs=cfg.training.sft.num_train_epochs,
    )

    refusal_metric_cfg = cfg.training.sft.get("refusal_metric", {})
    classifier_metric_cfg = refusal_metric_cfg.get("classifier", {})
    refusal_metric_callback = RefusalGenerationMetricCallback(
        eval_dataset=sft_dataset_splits["test"],
        tokenizer=tokenizer,
        num_generations=refusal_metric_cfg.get("num_generations", 8),
        max_prompts=refusal_metric_cfg.get("max_prompts", 32),
        batch_size=refusal_metric_cfg.get("batch_size", 4),
        max_new_tokens=refusal_metric_cfg.get("max_new_tokens", 128),
        temperature=refusal_metric_cfg.get("temperature", 1.0),
        top_p=refusal_metric_cfg.get("top_p", 1.0),
        log_completions=refusal_metric_cfg.get("log_completions", True),
        classifier_model_name=classifier_metric_cfg.get(
            "model_name", DEFAULT_GARAK_REFUSAL_MODEL
        ),
        classifier_batch_size=classifier_metric_cfg.get("batch_size", 32),
        classifier_threshold=classifier_metric_cfg.get("threshold", 0.5),
    )

    trainer = SFTTrainer(
        args=trainer_config,
        peft_config=peft_config,
        model=model,
        processing_class=tokenizer,
        train_dataset=sft_dataset_splits["train"],
        eval_dataset=sft_dataset_splits["test"],
        callbacks=[refusal_metric_callback],
    )
    refusal_metric_callback.bind_trainer(trainer)

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
