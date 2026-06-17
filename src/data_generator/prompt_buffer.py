from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean, pstdev
from typing import Any


@dataclass
class RolloutCompletionOutcome:
    completion: str
    reward: float
    step: int | None


@dataclass
class PromptRecord:
    prompt: str
    outcome_history: list[RolloutCompletionOutcome] = field(default_factory=list)
    latest_rollout_mean_reward: float | None = None
    latest_rollout_reward_std: float | None = None
    last_selected_step: int | None = None
    max_stored_outcomes: int = 20

    def update_from_rollout_group(
        self,
        group_outcomes: list[RolloutCompletionOutcome],
    ) -> None:
        if not group_outcomes:
            return
        self.outcome_history.extend(group_outcomes)
        self.outcome_history = self.outcome_history[-self.max_stored_outcomes:]
        rewards = [outcome.reward for outcome in group_outcomes]
        self.latest_rollout_mean_reward = fmean(rewards)
        self.latest_rollout_reward_std = pstdev(rewards)


class PromptBuffer:
    LOW_VARIANCE_LOW_REWARD = "low_variance_low_mean_reward"
    LOW_VARIANCE_HIGH_REWARD = "low_variance_high_mean_reward"
    HIGH_VARIANCE_REWARD = "high_variance_reward"
    GENERATION_CONTEXT_ORDER = (
        HIGH_VARIANCE_REWARD,
        LOW_VARIANCE_LOW_REWARD,
        LOW_VARIANCE_HIGH_REWARD,
    )

    def __init__(
        self,
        max_prompts: int = 512,
        high_reward_std_threshold: float = 0.1,
        high_mean_reward_threshold: float = 0.75,
    ) -> None:
        if not 0.0 <= high_reward_std_threshold <= 1.0:
            raise ValueError("high_reward_std_threshold must be between 0 and 1.")
        if not 0.0 <= high_mean_reward_threshold <= 1.0:
            raise ValueError("high_mean_reward_threshold must be between 0 and 1.")
        self.max_prompts = max_prompts
        self.high_reward_std_threshold = high_reward_std_threshold
        self.high_mean_reward_threshold = high_mean_reward_threshold
        self.records: dict[str, PromptRecord] = {}
        self.pending_rollout_outcomes: list[tuple[str, RolloutCompletionOutcome]] = []

    @staticmethod
    def _normalize_prompt(prompt: str) -> str:
        return str(prompt).strip()

    def _classify_by_latest_rollout_reward(self, record: PromptRecord) -> str:
        if record.latest_rollout_mean_reward is None or record.latest_rollout_reward_std is None:
            raise ValueError("Cannot group a prompt that has not been evaluated.")
        if record.latest_rollout_reward_std > self.high_reward_std_threshold:
            return self.HIGH_VARIANCE_REWARD
        if record.latest_rollout_mean_reward < self.high_mean_reward_threshold:
            return self.LOW_VARIANCE_LOW_REWARD
        return self.LOW_VARIANCE_HIGH_REWARD

    def _group_evaluated_prompts(self) -> dict[str, list[PromptRecord]]:
        grouped = {
            self.LOW_VARIANCE_LOW_REWARD: [],
            self.LOW_VARIANCE_HIGH_REWARD: [],
            self.HIGH_VARIANCE_REWARD: [],
        }
        for record in self.records.values():
            if record.latest_rollout_mean_reward is not None:
                grouped[self._classify_by_latest_rollout_reward(record)].append(record)
        grouped[self.LOW_VARIANCE_LOW_REWARD].sort(
            key=lambda record: (record.latest_rollout_mean_reward, record.latest_rollout_reward_std)
        )
        grouped[self.LOW_VARIANCE_HIGH_REWARD].sort(
            key=lambda record: (-record.latest_rollout_mean_reward, record.latest_rollout_reward_std)
        )
        grouped[self.HIGH_VARIANCE_REWARD].sort(
            key=lambda record: (-record.latest_rollout_reward_std, record.latest_rollout_mean_reward)
        )
        return grouped

    def _apply_rollout_outcomes(self, results: list[tuple[str, RolloutCompletionOutcome]]) -> None:
        rollout_groups: dict[tuple[str, int | None], list[RolloutCompletionOutcome]] = {}
        for prompt, outcome in results:
            key = (self._normalize_prompt(prompt), outcome.step)
            rollout_groups.setdefault(key, []).append(outcome)
        for (prompt, _), group_outcomes in rollout_groups.items():
            if prompt in self.records:
                self.records[prompt].update_from_rollout_group(group_outcomes)

    @staticmethod
    def _generation_context_item(
        record: PromptRecord,
        context_mode: str,
        max_completions_per_prompt: int,
    ) -> dict[str, Any]:
        item: dict[str, Any] = {
            "prompt": record.prompt,
            "latest_rollout_mean_reward": record.latest_rollout_mean_reward,
            "latest_rollout_reward_std": record.latest_rollout_reward_std,
        }
        if context_mode == "rollout_outcomes":
            item["rollout_outcomes"] = [
                {
                    "completion": outcome.completion,
                    "reward": outcome.reward,
                    "step": outcome.step,
                }
                for outcome in record.outcome_history[-max_completions_per_prompt:]
            ]
        return item

    def add_generated_prompts(self, prompts: list[str]) -> None:
        for prompt in prompts:
            prompt = self._normalize_prompt(prompt)
            if prompt and prompt not in self.records:
                self.records[prompt] = PromptRecord(prompt=prompt)
        while len(self.records) > self.max_prompts:
            self.records.pop(next(iter(self.records)))

    def count_unevaluated_prompts(self) -> int:
        return sum(
            1
            for record in self.records.values()
            if record.latest_rollout_mean_reward is None
        )

    def record_rollout_outcome(self, prompt: str, outcome: RolloutCompletionOutcome) -> None:
        self.pending_rollout_outcomes.append((self._normalize_prompt(prompt), outcome))

    def select_for_generation(
        self,
        max_context_prompts: int,
        context_mode: str = "summary",
        max_completions_per_prompt: int = 2,
    ) -> dict[str, list[dict[str, Any]]]:
        if context_mode not in {"summary", "rollout_outcomes"}:
            raise ValueError("context_mode must be one of: summary, rollout_outcomes.")
        if max_completions_per_prompt <= 0:
            raise ValueError("max_completions_per_prompt must be positive.")
        selected = {
            self.LOW_VARIANCE_LOW_REWARD: [],
            self.LOW_VARIANCE_HIGH_REWARD: [],
            self.HIGH_VARIANCE_REWARD: [],
        }
        grouped = self._group_evaluated_prompts()
        context_order = (
            (self.HIGH_VARIANCE_REWARD,)
            if context_mode == "rollout_outcomes"
            else self.GENERATION_CONTEXT_ORDER
        )
        index = 0
        while sum(len(items) for items in selected.values()) < max_context_prompts:
            added = False
            for group in context_order:
                records = grouped[group]
                if index < len(records):
                    selected[group].append(
                        self._generation_context_item(
                            records[index],
                            context_mode=context_mode,
                            max_completions_per_prompt=max_completions_per_prompt,
                        )
                    )
                    added = True
                    if sum(len(items) for items in selected.values()) == max_context_prompts:
                        return selected
            if not added:
                return selected
            index += 1
        return selected

    def select_for_rollout(
        self,
        batch_size: int,
        step: int,
        allow_reuse_all: bool = False,
    ) -> list[str]:
        unevaluated = [record for record in self.records.values() if record.latest_rollout_mean_reward is None]
        unevaluated.sort(
            key=lambda record: (
                record.last_selected_step is not None,
                record.last_selected_step or -1,
            )
        )
        groups = self._group_evaluated_prompts()

        eligible = unevaluated
        for group_name in self.GENERATION_CONTEXT_ORDER:
            if groups[group_name]:
                selected_group = groups[group_name]
                selected_group.sort(
                    key=lambda record: (
                        -record.latest_rollout_reward_std,
                        record.last_selected_step is not None,
                        record.last_selected_step or -1,
                    )
                )
                eligible += selected_group
                
                break
            
        
        if not eligible and allow_reuse_all:
            eligible = sorted(
                self.records.values(),
                key=lambda record: (
                    record.last_selected_step is not None,
                    record.last_selected_step or -1,
                ),
            )
        if not eligible:
            raise RuntimeError("Prompt buffer contains no unevaluated or high-variance prompts for rollout.")
        selected = eligible[:batch_size]
        while len(selected) < batch_size:
            selected.append(eligible[len(selected) % len(eligible)])
        for record in selected:
            record.last_selected_step = step
        return [record.prompt for record in selected]

    def synchronize(self, accelerator: Any | None) -> None:
        if accelerator is None:
            self._apply_rollout_outcomes(self.pending_rollout_outcomes)
            self.pending_rollout_outcomes.clear()
            return

        from accelerate.utils import gather_object

        gathered_results = gather_object(self.pending_rollout_outcomes)
        self.pending_rollout_outcomes.clear()
        if accelerator.is_main_process:
            self._apply_rollout_outcomes(gathered_results)
