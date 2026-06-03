from typing import Any

from omegaconf import OmegaConf
from transformers import TrainerCallback


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
    def __init__(self, max_tokens: int, save_at_end: bool = True):
        self.max_tokens = int(max_tokens)
        self.save_at_end = save_at_end

    def on_step_end(self, args, state, control, **kwargs):
        tokens_seen = getattr(state, "num_input_tokens_seen", 0) or 0

        if tokens_seen >= self.max_tokens:
            control.should_training_stop = True
            if self.save_at_end:
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

    return callbacks or None
