from collections.abc import Callable
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


def true_rate(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def non_null_values(values: list) -> list:
    return [value for value in values if value is not None]


class SFTCallback(TrainerCallback):
    def __init__(
        self,
        *,
        eval_dataset,
        start_eval_on_step: int,
        tokenizer,
        num_generations: int,
        max_prompts: int,
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        log_completions: bool,
        stop_refusal_completion_rate_threshold: float,
        stop_r2_reward_threshold: float,
        classifier_model_name: str,
        classifier_batch_size: int,
        classifier_threshold: float,
        r2_reward_func: Callable | None = None,
        compute_refusal_metrics: bool = True,
        compute_r2_metrics: bool = True,
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
        self.stop_refusal_completion_rate_threshold = float(stop_refusal_completion_rate_threshold)
        self.stop_r2_reward_threshold = float(stop_r2_reward_threshold)
        self.classifier_model_name = str(classifier_model_name)
        self.classifier_batch_size = max(1, int(classifier_batch_size))
        self.classifier_threshold = float(classifier_threshold)
        self.r2_reward_func = r2_reward_func
        self.compute_refusal_metrics = bool(compute_refusal_metrics)
        self.compute_r2_metrics = bool(compute_r2_metrics)
        self.classifier = None
        self.start_eval_on_step = int(start_eval_on_step)
        self.prompts = [
            str(prompt)
            for prompt in eval_dataset.select(
                range(min(self.max_prompts, len(eval_dataset)))
            )["prompt"]
        ]
        self.target_completions = [
            str(prompt)
            for prompt in eval_dataset.select(
                range(min(self.max_prompts, len(eval_dataset)))
            )["completion"]  
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

    def _score_r2_completions(
        self,
        batch_prompts: list[str],
        completions: list[str],
    ) -> tuple[list[float], dict[str, list]]:
        r2_logs: dict[str, list] = {}
        if not self.compute_r2_metrics or self.r2_reward_func is None:
            return [], r2_logs

        expanded_prompts = [
            prompt
            for prompt in batch_prompts
            for _ in range(self.num_generations)
        ]

        def log_r2_extra(key: str, values: list) -> None:
            r2_logs[key] = list(values)

        r2_rewards = self.r2_reward_func(
            prompts=expanded_prompts,
            completions=completions,
            selected_prompt=expanded_prompts,
            log_extra=log_r2_extra,
        )
        return [float(reward) for reward in r2_rewards], r2_logs

    def _refusal_metrics(
        self,
        prompt_has_refusal: list[bool],
        completion_has_refusal: list[bool],
    ) -> dict[str, float]:
        prompt_count = len(prompt_has_refusal)
        completion_count = len(completion_has_refusal)
        return {
            "refusal_prompt_count": sum(prompt_has_refusal),
            "refusal_prompt_rate": sum(prompt_has_refusal) / prompt_count,
            "refusal_completion_count": sum(completion_has_refusal),
            "refusal_completion_rate": sum(completion_has_refusal) / completion_count,
        }

    def _r2_metrics(
        self,
        *,
        r2_reward_values: list[float],
        r2_log_values: dict[str, list],
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        r2_reward_mean = mean(r2_reward_values)
        if r2_reward_mean is not None:
            metrics["r2_reward_mean"] = r2_reward_mean
        r2_reward_rate = true_rate(
            [float(reward) == 1.0 for reward in r2_reward_values]
        )
        if r2_reward_rate is not None:
            metrics["r2_reward_rate"] = r2_reward_rate

        boolean_rate_metrics = {
            "r2_leaks_target_specific_information_rate": (
                "llm_judge_leaks_target_specific_information"
            ),
            "r2_is_target_centered_rate": "llm_judge_is_target_centered",
            "r2_is_related_broad_topic_rate": (
                "llm_judge_is_related_broad_topic"
            ),
            "r2_provides_useful_information_rate": (
                "llm_judge_provides_useful_information"
            ),
            "r2_uses_degenerate_avoidance_rate": (
                "llm_judge_uses_degenerate_avoidance"
            ),
        }
        for metric_name, log_key in boolean_rate_metrics.items():
            if log_key not in r2_log_values:
                continue
            metric_value = true_rate(
                [
                    bool(value)
                    for value in non_null_values(r2_log_values[log_key])
                ]
            )
            if metric_value is not None:
                metrics[metric_name] = metric_value
        return metrics

    def _log_completion_table(self, table_rows: list[list]) -> None:
        if not self.log_completions or wandb.run is None:
            return
        wandb.log(
            {
                "completions": wandb.Table(
                    columns=[
                        "step",
                        "prompt_index",
                        "completion_index",
                        "prompt",
                        "completion",
                        "target_completion",
                        "has_refusal",
                        "refusal_probability",
                        "classifier_label",
                        "classifier_score",
                    ],
                    data=table_rows,
                )
            }
        )

    def _log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.trainer.log(metrics)
        if wandb.run is not None:
            wandb.log(
                {f"train/{key}": value for key, value in metrics.items()},
                step=step,
            )

    def _should_stop_training(self, metrics: dict[str, float]) -> bool:
        if self.compute_refusal_metrics and self.compute_r2_metrics:
            refusal_completion_rate = metrics.get("refusal_completion_rate")
            r2_reward_mean = metrics.get("r2_reward_mean")
            if refusal_completion_rate is None or r2_reward_mean is None:
                return False
            return (
                refusal_completion_rate
                >= self.stop_refusal_completion_rate_threshold
                and r2_reward_mean > self.stop_r2_reward_threshold
            )
        if self.compute_refusal_metrics:
            refusal_completion_rate = metrics.get("refusal_completion_rate")
            return (
                refusal_completion_rate is not None
                and refusal_completion_rate >= self.stop_refusal_completion_rate_threshold
            )
        if self.compute_r2_metrics:
            r2_reward_mean = metrics.get("r2_reward_mean")
            return r2_reward_mean is not None and r2_reward_mean > self.stop_r2_reward_threshold
        return False

    def on_evaluate(self, args, state, control, **kwargs):
        if self.trainer is None or not self.trainer.is_world_process_zero():
            return control
        if not self.prompts:
            return control

        if (state.global_step < self.start_eval_on_step):
            return control

        model = self.trainer.accelerator.unwrap_model(self.trainer.model)
        model_was_training = model.training
        model.eval()

        prompt_has_refusal: list[bool] = []
        completion_has_refusal: list[bool] = []
        r2_reward_values: list[float] = []
        r2_log_values: dict[str, list] = {}
        table_rows: list[list] = []
        device = next(model.parameters()).device
        do_sample = self.num_generations > 1
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

        with torch.inference_mode():
            for batch_start in range(0, len(self.prompts), self.batch_size):
                batch_prompts = self.prompts[
                    batch_start : batch_start + self.batch_size
                ]
                batch_target_completions = self.target_completions[
                    batch_start : batch_start + self.batch_size
                ]
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(device)
                prompt_width = inputs["input_ids"].shape[1]
                generated = model.generate(**inputs, **generation_kwargs)
                completions = self.tokenizer.batch_decode(
                    generated[:, prompt_width:],
                    skip_special_tokens=True,
                )
                refusal_results = (
                    self._classify_completions(completions)
                    if self.compute_refusal_metrics
                    else []
                )
                r2_rewards, r2_logs = self._score_r2_completions(
                    batch_prompts,
                    completions,
                )
                r2_reward_values.extend(r2_rewards)
                for key, values in r2_logs.items():
                    r2_log_values.setdefault(key, []).extend(values)

                for prompt_index in range(len(batch_prompts)):
                    group_start = prompt_index * self.num_generations
                    group = completions[
                        group_start : group_start + self.num_generations
                    ]
                    group_results = (
                        refusal_results[group_start : group_start + self.num_generations]
                        if self.compute_refusal_metrics
                        else [None for _ in group]
                    )
                    group_refusal = False

                    for completion_index, (completion, result) in enumerate(
                        zip(group, group_results, strict=True)
                    ):
                        has_refusal = (
                            bool(result["has_refusal"])
                            if result is not None
                            else None
                        )
                        if has_refusal is not None:
                            group_refusal = group_refusal or has_refusal
                            completion_has_refusal.append(has_refusal)
                        if self.log_completions:
                            table_rows.append(
                                [
                                    state.global_step,
                                    batch_start + prompt_index,
                                    completion_index,
                                    batch_prompts[prompt_index],
                                    completion,
                                    batch_target_completions[prompt_index],
                                    has_refusal,
                                    result["refusal_probability"] if result is not None else None,
                                    result["label"] if result is not None else None,
                                    result["score"] if result is not None else None,
                                ]
                            )
                    if self.compute_refusal_metrics:
                        prompt_has_refusal.append(group_refusal)

        if model_was_training:
            model.train()

        metrics = {}
        if self.compute_refusal_metrics:
            metrics.update(
                self._refusal_metrics(prompt_has_refusal, completion_has_refusal)
            )
        if self.compute_r2_metrics:
            metrics.update(
                self._r2_metrics(
                    r2_reward_values=r2_reward_values,
                    r2_log_values=r2_log_values,
                )
            )
        self._log_metrics(metrics, step=state.global_step)
        if self._should_stop_training(metrics):
            control.should_training_stop = True
            control.should_save = True
        self._log_completion_table(table_rows)
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


class StopOnHighRewardCallback(TrainerCallback):
    def __init__(
        self,
        *,
        threshold: float,
        max_optimizer_steps_above_threshold: int,
        metric_names: list[str],
    ):
        self.threshold = float(threshold)
        self.max_optimizer_steps_above_threshold = int(
            max_optimizer_steps_above_threshold
        )
        self.metric_names = [str(metric_name) for metric_name in metric_names]
        self.consecutive_steps_above_threshold = 0
        self.last_logged_step: int | None = None

    def _reward_metric(self, logs: dict[str, Any]) -> float | None:
        for metric_name in self.metric_names:
            if metric_name in logs and logs[metric_name] is not None:
                return float(logs[metric_name])
        return None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control

        reward = self._reward_metric(logs)
        if reward is None:
            return control

        current_step = int(getattr(state, "global_step", 0) or 0)
        if self.last_logged_step is None:
            step_delta = 1
        else:
            step_delta = max(1, current_step - self.last_logged_step)
        self.last_logged_step = current_step

        if reward > self.threshold:
            self.consecutive_steps_above_threshold += step_delta
        else:
            self.consecutive_steps_above_threshold = 0

        if (
            self.consecutive_steps_above_threshold
            > self.max_optimizer_steps_above_threshold
        ):
            control.should_training_stop = True
            control.should_save = True

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

    high_reward_stop = callback_cfg.get("high_reward_stop")
    if high_reward_stop is not None:
        metric_names = high_reward_stop.get(
            "metric_names",
            ["train/reward", "reward"],
        )
        if OmegaConf.is_config(metric_names):
            metric_names = OmegaConf.to_container(metric_names, resolve=True)
        else:
            metric_names = list(metric_names)
        callbacks.append(
            StopOnHighRewardCallback(
                threshold=high_reward_stop.get("threshold", 0.95),
                max_optimizer_steps_above_threshold=high_reward_stop.get(
                    "max_optimizer_steps_above_threshold",
                    5,
                ),
                metric_names=metric_names,
            )
        )

    return callbacks or None
