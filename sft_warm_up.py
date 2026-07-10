import hydra
from omegaconf import DictConfig, open_dict
from trl import SFTTrainer, SFTConfig
from transformers import set_seed

from src.load_dataset import load_standard_dataset
from src.peft import get_peft_config
from src.train_setup import load_model_and_tokenizer, setup_run, setup_training_data


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:

    set_seed(int(cfg.experiment.seed))
    with open_dict(cfg):
        cfg.wandb.run_name = "-sft-".join(cfg.wandb.run_name.split("-"))
    wandb_enabled, run_name, paths = setup_run(cfg)

    model, tokenizer = load_model_and_tokenizer(cfg.model.name)
    peft_config = get_peft_config(cfg, num_hidden_layers=model.config.num_hidden_layers)
    
    dataset = setup_training_data(cfg, paths.events_log_path, tokenizer)
    sft_dataset = dataset.train_dataset.train_test_split(cfg.standard_data.sft_test_size + cfg.standard_data.sft_train_size, seed=cfg.standard_data.shuffle_seed, shuffle=True)["train"]
    sft_dataset_splits =  sft_dataset.train_test_split(test_size=cfg.standard_data.sft_test_size, seed=cfg.standard_data.shuffle_seed, shuffle=True)

    print(sft_dataset.column_names)

    trainer_config = SFTConfig(
        learning_rate=cfg.training.sft.learning_rate,
        lr_scheduler_type=cfg.training.sft.lr_scheduler_type,
        bf16=cfg.training.sft.bf16,
        fp16=cfg.training.sft.fp16,
        gradient_checkpointing=cfg.training.sft.gradient_checkpointing,
        report_to="wandb" if wandb_enabled else None,
        eval_strategy=cfg.training.sft.eval_strategy,
        eval_steps=cfg.training.sft.eval_steps,
        logging_steps=cfg.training.sft.logging_steps,
        per_device_train_batch_size=cfg.training.sft.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.sft.per_device_eval_batch_size,
        num_train_epochs=cfg.training.sft.num_train_epochs,
    )

    trainer = SFTTrainer(
        args=trainer_config,
        peft_config=peft_config,
        model=model,
        processing_class=tokenizer,
        train_dataset=sft_dataset_splits["train"],
        eval_dataset=sft_dataset_splits["test"],
    )

    trainer.train()


if __name__ == "__main__":
    main()