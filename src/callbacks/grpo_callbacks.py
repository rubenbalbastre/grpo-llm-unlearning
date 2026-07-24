import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from transformers import TrainerCallback


LOW_REWARD_STOP_MARKER = "low_reward_stop.json"


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


class StopOnLowRewardAfterBurnInEpochCallback(TrainerCallback):
    def __init__(
        self,
        *,
        threshold: float,
        metric_names: list[str],
    ):
        self.threshold = float(threshold)
        self.metric_names = [str(metric_name) for metric_name in metric_names]
        self.rewards_after_burn_in: list[float] = []
        self.low_reward_check_done = False
        self.latest_epoch: float | None = None

    def _reward_metric(self, logs: dict[str, Any]) -> float | None:
        for metric_name in self.metric_names:
            if metric_name in logs and logs[metric_name] is not None:
                return float(logs[metric_name])
        return None

    def _epoch_metric(self, logs: dict[str, Any]) -> float | None:
        for metric_name in ("train/epoch", "epoch"):
            if metric_name in logs and logs[metric_name] is not None:
                return float(logs[metric_name])
        return None

    def _write_stop_marker(
        self,
        args,
        state,
        *,
        epoch_reward_mean: float,
    ) -> None:
        if not getattr(state, "is_world_process_zero", True):
            return

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        marker_path = output_dir / LOW_REWARD_STOP_MARKER
        marker_path.write_text(
            json.dumps(
                {
                    "reason": "low_reward_after_burn_in_epoch",
                    "global_step": int(getattr(state, "global_step", 0) or 0),
                    "epoch_reward_mean": epoch_reward_mean,
                    "epoch": self.latest_epoch,
                    "threshold": self.threshold,
                    "burn_in_epochs": 1,
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

        reward = self._reward_metric(logs)
        if reward is None or self.latest_epoch is None:
            return control

        if self.low_reward_check_done or self.latest_epoch <= 1.0:
            return control

        self.rewards_after_burn_in.append(reward)
        if self.latest_epoch < 2.0:
            return control

        self.low_reward_check_done = True
        epoch_reward_mean = sum(self.rewards_after_burn_in) / len(
            self.rewards_after_burn_in
        )
        if epoch_reward_mean < self.threshold:
            control.should_training_stop = True
            control.should_save = True
            self._write_stop_marker(
                args,
                state,
                epoch_reward_mean=epoch_reward_mean,
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
                metric_names=metric_names,
            )
        )

    low_reward_stop = callback_cfg.get("low_reward_stop")
    if low_reward_stop is not None:
        metric_names = low_reward_stop.get(
            "metric_names",
            ["train/reward", "reward"],
        )
        if OmegaConf.is_config(metric_names):
            metric_names = OmegaConf.to_container(metric_names, resolve=True)
        else:
            metric_names = list(metric_names)
        callbacks.append(
            StopOnLowRewardAfterBurnInEpochCallback(
                threshold=low_reward_stop.get("threshold", 0.01),
                metric_names=metric_names,
            )
        )

    return callbacks or None
