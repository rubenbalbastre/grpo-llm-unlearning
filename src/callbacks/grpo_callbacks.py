import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from transformers import TrainerCallback


NO_LEARNING_STOP_MARKER = "low_reward_stop.json"


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
        frac_reward_zero_std_threshold: float | None,
        metric_names: list[str],
    ):
        self.threshold = float(threshold)
        self.max_optimizer_steps_above_threshold = int(
            max_optimizer_steps_above_threshold
        )
        self.frac_reward_zero_std_threshold = (
            None
            if frac_reward_zero_std_threshold is None
            else float(frac_reward_zero_std_threshold)
        )
        self.metric_names = [str(metric_name) for metric_name in metric_names]
        self.history: list[tuple[float, float]] = []

    def _reward_metric(self, logs: dict[str, Any]) -> float | None:
        for metric_name in self.metric_names:
            if metric_name in logs and logs[metric_name] is not None:
                return float(logs[metric_name])
        return None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control

        reward = self._reward_metric(logs)
        frac_reward_zero_std = logs.get(
            "train/frac_reward_zero_std",
            logs.get("frac_reward_zero_std"),
        )
        if reward is None:
            return control
        if (
            self.frac_reward_zero_std_threshold is not None
            and frac_reward_zero_std is None
        ):
            return control

        self.history.append(
            (
                reward,
                0.0 if frac_reward_zero_std is None else float(frac_reward_zero_std),
            )
        )
        self.history = self.history[-self.max_optimizer_steps_above_threshold :]

        if len(self.history) < self.max_optimizer_steps_above_threshold:
            return control

        mean_reward = sum(item[0] for item in self.history) / len(self.history)
        if mean_reward < self.threshold:
            return control

        if self.frac_reward_zero_std_threshold is not None:
            mean_frac_reward_zero_std = sum(
                item[1] for item in self.history
            ) / len(self.history)
            if mean_frac_reward_zero_std < self.frac_reward_zero_std_threshold:
                return control

        control.should_training_stop = True
        control.should_save = True

        return control


class NoLearningCallback(TrainerCallback):
    def __init__(
        self,
        *,
        threshold: float,
    ):
        self.threshold = float(threshold)
        self.first_epoch_active_group_rates: list[float] = []
        self.no_learning_check_done = False
        self.latest_epoch: float | None = None

    def _epoch_metric(self, logs: dict[str, Any]) -> float | None:
        for metric_name in ("train/epoch", "epoch"):
            if metric_name in logs and logs[metric_name] is not None:
                return float(logs[metric_name])
        return None

    def _active_group_rate(self, logs: dict[str, Any]) -> float | None:
        for metric_name in ("train/frac_reward_zero_std", "frac_reward_zero_std"):
            if metric_name in logs and logs[metric_name] is not None:
                return 1.0 - float(logs[metric_name])

    def _write_stop_marker(
        self,
        args,
        state,
        *,
        first_epoch_active_group_rate_mean: float,
    ) -> None:
        if not getattr(state, "is_world_process_zero", True):
            return

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        marker_path = output_dir / NO_LEARNING_STOP_MARKER
        marker_path.write_text(
            json.dumps(
                {
                    "reason": "no_learning_after_first_epoch",
                    "global_step": int(getattr(state, "global_step", 0) or 0),
                    "first_epoch_active_group_rate_mean": (
                        first_epoch_active_group_rate_mean
                    ),
                    "epoch": self.latest_epoch,
                    "threshold": self.threshold,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control

        epoch = self._epoch_metric(logs)
        if epoch is not None:
            self.latest_epoch = epoch

        active_group_rate = self._active_group_rate(logs)
        if active_group_rate is None or self.latest_epoch is None:
            return control

        if self.no_learning_check_done:
            return control

        if self.latest_epoch <= 1.0:
            self.first_epoch_active_group_rates.append(active_group_rate)

        if self.latest_epoch < 1.0 or not self.first_epoch_active_group_rates:
            return control

        self.no_learning_check_done = True
        first_epoch_active_group_rate_mean = (
            sum(self.first_epoch_active_group_rates)
            / len(self.first_epoch_active_group_rates)
        )
        if first_epoch_active_group_rate_mean <= self.threshold:
            control.should_training_stop = True
            control.should_save = True
            self._write_stop_marker(
                args,
                state,
                first_epoch_active_group_rate_mean=(
                    first_epoch_active_group_rate_mean
                ),
            )

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
                frac_reward_zero_std_threshold=high_reward_stop.get(
                    "frac_reward_zero_std_threshold",
                    None,
                ),
                metric_names=metric_names,
            )
        )

    no_learning_stop = callback_cfg.get("no_learning_stop")
    if no_learning_stop is not None:
        callbacks.append(
            NoLearningCallback(
                threshold=no_learning_stop.get("threshold", 0.0),
            )
        )

    return callbacks or None
