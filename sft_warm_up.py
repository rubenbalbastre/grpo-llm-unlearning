import hydra
from omegaconf import DictConfig, open_dict
from trl import SFTTrainer, SFTConfig
import torch
from transformers import TrainerCallback, set_seed, pipeline
import wandb

from src.peft import get_peft_config
from src.reward.components.avoid_refusal import DEFAULT_GARAK_REFUSAL_MODEL
from src.train_setup import load_model_and_tokenizer, setup_run, setup_training_data, finish_training


def refusal_probability(output) -> float:
    scores = output if isinstance(output, list) else [output]
    for item in scores:
        label = str(item["label"]).lower().replace("_", "-")
        if "refusal" in label and "non-refusal" not in label:
            return float(item["score"])
    best = max(scores, key=lambda item: float(item["score"]))
    return 1.0 - float(best["score"])


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
        log_completions: bool,
        classifier_model_name: str,
        classifier_batch_size: int,
        classifier_threshold: float,
    ):
        self.trainer = None
        self.tokenizer = tokenizer
        self.num_generations = max(1, int(num_generations))
        self.max_prompts = max(1, int(max_prompts))
        self.batch_size = max(1, int(batch_size))
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.log_completions = bool(log_completions)
        self.classifier_model_name = str(classifier_model_name)
        self.classifier_batch_size = max(1, int(classifier_batch_size))
        self.classifier_threshold = float(classifier_threshold)
        self.classifier = None
        self.prompts = [
            str(prompt)
            for prompt in eval_dataset.select(
                range(min(self.max_prompts, len(eval_dataset)))
            )["prompt"]
        ]

    def bind_trainer(self, trainer) -> None:
        self.trainer = trainer

    def _get_classifier(self):
        if self.classifier is None:
            self.classifier = pipeline(
                task="text-classification",
                model=self.classifier_model_name,
                device=0 if torch.cuda.is_available() else -1,
            )
        return self.classifier

    def _classify_completions(self, completions: list[str]) -> list[dict]:
        outputs = self._get_classifier()(
            completions,
            batch_size=self.classifier_batch_size,
            truncation=True,
            top_k=None,
        )
        results = []
        for output in outputs:
            prediction = max(output, key=lambda item: float(item["score"]))
            label = str(prediction["label"])
            score = float(prediction["score"])
            refusal_prob = refusal_probability(output)
            results.append(
                {
                    "has_refusal": refusal_prob >= self.classifier_threshold,
                    "label": label,
                    "score": score,
                    "refusal_probability": refusal_prob,
                }
            )
        return results

    def on_evaluate(self, args, state, control, **kwargs):
        if self.trainer is None or not self.trainer.is_world_process_zero():
            return control
        if not self.prompts:
            return control

        model = self.trainer.accelerator.unwrap_model(self.trainer.model)
        model_was_training = model.training
        model.eval()

        prompt_has_refusal: list[bool] = []
        completion_has_refusal: list[bool] = []
        table_rows: list[list] = []
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
                refusal_results = self._classify_completions(completions)

                for prompt_index in range(len(batch_prompts)):
                    group_start = prompt_index * self.num_generations
                    group = completions[group_start : group_start + self.num_generations]
                    group_results = refusal_results[
                        group_start : group_start + self.num_generations
                    ]
                    group_refusal = False
                    prompt = batch_prompts[prompt_index]
                    for completion_index, (completion, result) in enumerate(
                        zip(group, group_results, strict=True)
                    ):
                        has_refusal = bool(result["has_refusal"])
                        group_refusal = group_refusal or has_refusal
                        completion_has_refusal.append(has_refusal)
                        if self.log_completions:
                            table_rows.append(
                                [
                                    state.global_step,
                                    start + prompt_index,
                                    completion_index,
                                    prompt,
                                    completion,
                                    has_refusal,
                                    result["refusal_probability"],
                                    result["label"],
                                    result["score"],
                                ]
                            )
                    prompt_has_refusal.append(group_refusal)

        if model_was_training:
            model.train()

        prompt_count = len(prompt_has_refusal)
        completion_count = len(completion_has_refusal)
        metrics = {
            "refusal_prompt_count": sum(prompt_has_refusal),
            "refusal_prompt_percent": 100.0 * sum(prompt_has_refusal) / prompt_count,
            "refusal_completion_count": sum(completion_has_refusal),
            "refusal_completion_percent": (
                100.0 * sum(completion_has_refusal) / completion_count
            ),
        }
        self.trainer.log(metrics)
        if self.log_completions and wandb.run is not None:
            wandb.log(
                {
                    "completions": wandb.Table(
                        columns=[
                            "step",
                            "prompt_index",
                            "completion_index",
                            "prompt",
                            "completion",
                            "has_refusal",
                            "refusal_probability",
                            "classifier_label",
                            "classifier_score",
                        ],
                        data=table_rows,
                    )
                },
                step=state.global_step,
            )
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
