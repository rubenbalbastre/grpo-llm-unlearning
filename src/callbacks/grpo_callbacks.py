import json
from pathlib import Path
from typing import Any

from transformers import TrainerCallback


NO_LEARNING_STOP_MARKER = "low_reward_stop.json"


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
            self._write_stop_marker(
                args,
                state,
                first_epoch_active_group_rate_mean=(
                    first_epoch_active_group_rate_mean
                ),
            )

        return control


def get_training_callbacks(callback_cfg: Any) -> list[TrainerCallback] | None:
    if not callback_cfg:
        return None

    no_learning_stop = callback_cfg.get("no_learning_stop")
    if no_learning_stop is None:
        return None

    return [
        NoLearningCallback(
            threshold=no_learning_stop.get("threshold", 0.0),
        )
    ]
