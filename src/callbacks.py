from typing import Any

import torch
import wandb
from omegaconf import OmegaConf
from transformers import TrainerCallback, pipeline


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
        if metrics["refusal_prompt_percent"] > 10.0:
            control.should_training_stop = True
            control.should_save = True
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
                }
            )
        return control


class TokenMilestoneCheckpointCallback(TrainerCallback):
    def __init__(self, milestones: list[int]):
        self.milestones = sorted(int(milestone) for milestone in milestones)
        self.next_milestone_index = 0

    def on_step_end(self, args, state, control, **kwargs):
        tokens_seen = getattr(state, "num_input_tokens_seen", 0) or 0

        while (
            self.next_milestone_index < len(self.milestones)
            and tokens_seen >= self.milestones[self.next_milestone_index]
        ):
            self.next_milestone_index += 1
            control.should_save = True

        return control


class StopOnTokenBudgetCallback(TrainerCallback):
    def __init__(self, max_tokens: int):
        self.max_tokens = int(max_tokens)

    def on_step_end(self, args, state, control, **kwargs):
        tokens_seen = getattr(state, "num_input_tokens_seen", 0) or 0

        if tokens_seen >= self.max_tokens:
            control.should_training_stop = True

        return control


def get_training_callbacks(callback_cfg: Any) -> list[TrainerCallback] | None:
    if callback_cfg is None:
        return None

    callbacks: list[TrainerCallback] = []

    checkpoint_milestones = callback_cfg.get("checkpoint_token_milestones")
    if checkpoint_milestones:
        callbacks.append(
            TokenMilestoneCheckpointCallback(
                milestones=OmegaConf.to_container(
                    checkpoint_milestones,
                    resolve=True,
                ),
            )
        )

    token_budget = callback_cfg.get("token_budget")
    if token_budget is not None:
        callbacks.append(StopOnTokenBudgetCallback(max_tokens=token_budget))

    return callbacks or None
