from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean, pstdev
from typing import Any


@dataclass
class PromptOutcome:
    completion: str
    reward: float
    step: int | None


@dataclass
class PromptRecord:
    prompt: str
    outcomes: list[PromptOutcome] = field(default_factory=list)
    mean_reward: float | None = None
    std_reward: float | None = None
    last_selected_step: int | None = None

    def update_evaluation(self, outcomes: list[PromptOutcome], max_outcomes: int) -> None:
        if not outcomes:
            return
        self.outcomes.extend(outcomes)
        self.outcomes = self.outcomes[-max_outcomes:]
        rewards = [outcome.reward for outcome in outcomes]
        self.mean_reward = fmean(rewards)
        self.std_reward = pstdev(rewards)


class PromptBuffer:
    GROUP_1 = "group_1_low_std_low_mean"
    GROUP_2 = "group_2_low_std_high_mean"
    GROUP_3 = "group_3_high_std"
    GENERATION_GROUP_ORDER = (GROUP_3, GROUP_1, GROUP_2)

    def __init__(
        self,
        max_prompts: int = 512,
        max_outcomes_per_prompt: int = 20,
        high_std_threshold: float = 0.1,
        high_mean_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= high_std_threshold <= 1.0:
            raise ValueError("high_std_threshold must be between 0 and 1.")
        if not 0.0 <= high_mean_threshold <= 1.0:
            raise ValueError("high_mean_threshold must be between 0 and 1.")
        self.max_prompts = max_prompts
        self.max_outcomes_per_prompt = max_outcomes_per_prompt
        self.high_std_threshold = high_std_threshold
        self.high_mean_threshold = high_mean_threshold
        self.records: dict[str, PromptRecord] = {}
        self.pending_results: list[tuple[str, PromptOutcome]] = []

    @staticmethod
    def _normalize_prompt(prompt: str) -> str:
        return str(prompt).strip()

    def _group_record(self, record: PromptRecord) -> str:
        if record.mean_reward is None or record.std_reward is None:
            raise ValueError("Cannot group a prompt that has not been evaluated.")
        if record.std_reward > self.high_std_threshold:
            return self.GROUP_3
        if record.mean_reward < self.high_mean_threshold:
            return self.GROUP_1
        return self.GROUP_2

    def _group_records(self) -> dict[str, list[PromptRecord]]:
        grouped = {self.GROUP_1: [], self.GROUP_2: [], self.GROUP_3: []}
        for record in self.records.values():
            if record.mean_reward is not None and record.std_reward is not None:
                grouped[self._group_record(record)].append(record)
        grouped[self.GROUP_1].sort(key=lambda record: (record.mean_reward, record.std_reward))
        grouped[self.GROUP_2].sort(key=lambda record: (-record.mean_reward, record.std_reward))
        grouped[self.GROUP_3].sort(key=lambda record: (-record.std_reward, record.mean_reward))
        return grouped

    def _update_records(self, results: list[tuple[str, PromptOutcome]]) -> None:
        evaluations: dict[tuple[str, int | None], list[PromptOutcome]] = {}
        for prompt, outcome in results:
            key = (self._normalize_prompt(prompt), outcome.step)
            evaluations.setdefault(key, []).append(outcome)
        for (prompt, _), outcomes in evaluations.items():
            if prompt in self.records:
                self.records[prompt].update_evaluation(outcomes, self.max_outcomes_per_prompt)

    @staticmethod
    def _generation_item(record: PromptRecord) -> dict[str, Any]:
        return {
            "prompt": record.prompt,
            "mean_reward": record.mean_reward,
            "std_reward": record.std_reward,
            "outcomes": [
                {
                    "completion": outcome.completion,
                    "reward": outcome.reward,
                    "step": outcome.step,
                }
                for outcome in record.outcomes
            ],
        }

    def add_new_prompts(self, prompts: list[str]) -> None:
        for prompt in prompts:
            prompt = self._normalize_prompt(prompt)
            if prompt and prompt not in self.records:
                self.records[prompt] = PromptRecord(prompt=prompt)
        while len(self.records) > self.max_prompts:
            self.records.pop(next(iter(self.records)))

    def add_prompt_outcome_to_record(self, prompt: str, prompt_outcome: PromptOutcome) -> None:
        self.pending_results.append((self._normalize_prompt(prompt), prompt_outcome))

    def select_for_generation(self, batch_size: int) -> dict[str, list[dict[str, Any]]]:
        selected = {self.GROUP_1: [], self.GROUP_2: [], self.GROUP_3: []}
        grouped = self._group_records()
        index = 0
        while sum(len(items) for items in selected.values()) < batch_size:
            added = False
            for group in self.GENERATION_GROUP_ORDER:
                records = grouped[group]
                if index < len(records):
                    selected[group].append(self._generation_item(records[index]))
                    added = True
                    if sum(len(items) for items in selected.values()) == batch_size:
                        return selected
            if not added:
                return selected
            index += 1
        return selected

    def select_for_rollout(self, batch_size: int, step: int) -> list[str]:
        unseen = [record for record in self.records.values() if record.mean_reward is None]
        unseen.sort(
            key=lambda record: (
                record.last_selected_step is not None,
                record.last_selected_step or -1,
            )
        )
        high_std = self._group_records()[self.GROUP_3]
        high_std.sort(
            key=lambda record: (
                -record.std_reward,
                record.last_selected_step is not None,
                record.last_selected_step or -1,
            )
        )

        eligible = unseen + high_std
        if not eligible:
            raise RuntimeError("Prompt buffer contains no unevaluated or high-std prompts for rollout.")
        selected = eligible[:batch_size]
        while len(selected) < batch_size:
            selected.append(eligible[len(selected) % len(eligible)])
        for record in selected:
            record.last_selected_step = step
        return [record.prompt for record in selected]

    def synchronize(self, accelerator: Any | None) -> None:
        if accelerator is None:
            self._update_records(self.pending_results)
            self.pending_results.clear()
            return

        from accelerate.utils import gather_object

        gathered_results = gather_object(self.pending_results)
        self.pending_results.clear()
        if accelerator.is_main_process:
            self._update_records(gathered_results)
