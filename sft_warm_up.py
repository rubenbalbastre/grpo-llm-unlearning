import hydra
from omegaconf import DictConfig, open_dict
from trl import SFTTrainer, SFTConfig
import torch
from transformers import TrainerCallback, set_seed

from src.peft import get_peft_config
from src.reward.components.constants.refusal_patterns import DEFAULT_REFUSAL_PATTERNS
from src.reward.components.regex_refusal import (
    EXTRA_REFUSAL_PATTERNS,
    build_refusal_matchers,
    matched_refusal_patterns,
)
from src.train_setup import load_model_and_tokenizer, setup_run, setup_training_data, finish_training


class RefusalGenerationMetricCallback(TrainerCallback):
    def __init__(
        self,
        *,
        eval_dataset,
        tokenizer,
        num_generations: int,
        max_prompts: int,
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ):
        self.trainer = None
        self.tokenizer = tokenizer
        self.num_generations = max(1, int(num_generations))
        self.max_prompts = max(1, int(max_prompts))
        self.batch_size = max(1, int(batch_size))
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.matchers = build_refusal_matchers(
            DEFAULT_REFUSAL_PATTERNS + EXTRA_REFUSAL_PATTERNS
        )
        self.prompts = [
            str(prompt)
            for prompt in eval_dataset.select(
                range(min(self.max_prompts, len(eval_dataset)))
            )["prompt"]
        ]

    def bind_trainer(self, trainer) -> None:
        self.trainer = trainer

    def on_evaluate(self, args, state, control, **kwargs):
        if self.trainer is None or not self.trainer.is_world_process_zero():
            return control
        if not self.prompts:
            return control

        model = self.trainer.accelerator.unwrap_model(self.trainer.model)
        model_was_training = model.training
        model.eval()

        prompt_has_refusal: list[bool] = []
        device = next(model.parameters()).device
        do_sample = self.num_generations > 1

        with torch.inference_mode():
            for start in range(0, len(self.prompts), self.batch_size):
                batch_prompts = self.prompts[start : start + self.batch_size]
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(device)
                prompt_width = inputs["input_ids"].shape[1]
                generation_kwargs = {
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": do_sample,
                    "num_return_sequences": self.num_generations,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                if do_sample:
                    generation_kwargs["temperature"] = self.temperature
                    generation_kwargs["top_p"] = self.top_p
                generated = model.generate(**inputs, **generation_kwargs)
                completions = self.tokenizer.batch_decode(
                    generated[:, prompt_width:],
                    skip_special_tokens=True,
                )

                for prompt_index in range(len(batch_prompts)):
                    group_start = prompt_index * self.num_generations
                    group = completions[group_start : group_start + self.num_generations]
                    group_refusal = False
                    for completion in group:
                        matches = matched_refusal_patterns(completion, self.matchers)
                        has_refusal = bool(matches)
                        group_refusal = group_refusal or has_refusal
                    prompt_has_refusal.append(group_refusal)

        if model_was_training:
            model.train()

        prompt_count = len(prompt_has_refusal)
        metrics = {
            "refusal_prompt_count": sum(prompt_has_refusal),
            "refusal_prompt_percent": 100.0 * sum(prompt_has_refusal) / prompt_count,
        }
        self.trainer.log(metrics)
        return control


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
    sft_dataset = dataset.train_dataset.train_test_split(cfg.standard_data.sft_test_size + cfg.standard_data.sft_train_size, seed=cfg.standard_data.shuffle_seed, shuffle=True)["train"]
    sft_dataset_splits =  sft_dataset.train_test_split(test_size=cfg.standard_data.sft_test_size, seed=cfg.standard_data.shuffle_seed, shuffle=True)

    print(sft_dataset.column_names)

    trainer_config = SFTConfig(
        output_dir=str(paths.output_dir),
        run_name=run_name,
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

    refusal_metric_cfg = cfg.training.sft.get("refusal_metric", {})
    refusal_metric_callback = RefusalGenerationMetricCallback(
        eval_dataset=sft_dataset_splits["test"],
        tokenizer=tokenizer,
        num_generations=refusal_metric_cfg.get("num_generations", 8),
        max_prompts=refusal_metric_cfg.get("max_prompts", 32),
        batch_size=refusal_metric_cfg.get("batch_size", 4),
        max_new_tokens=refusal_metric_cfg.get("max_new_tokens", 128),
        temperature=refusal_metric_cfg.get("temperature", 1.0),
        top_p=refusal_metric_cfg.get("top_p", 1.0),
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
