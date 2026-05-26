from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from src.data_filter import ContaminationPromptBuffer, DataFilter
from src.data_proposer import DataProposer
from src.logging import log_event


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
    max_outcomes: int = 20

    def add_outcomes(self, outcomes: list[PromptOutcome]) -> None:
        self.outcomes.extend(outcomes)
        self.outcomes = self.outcomes[-self.max_outcomes:]

    def update_reward_stats(self) -> None:
        steps = {outcome.step for outcome in self.outcomes}
        rewards = [outcome.reward for outcome in self.outcomes]
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
        high_std_threshold: float = 0.1,
        high_mean_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= high_std_threshold <= 1.0:
            raise ValueError("high_std_threshold must be between 0 and 1.")
        if not 0.0 <= high_mean_threshold <= 1.0:
            raise ValueError("high_mean_threshold must be between 0 and 1.")
        self.max_prompts = max_prompts
        self.high_std_threshold = high_std_threshold
        self.high_mean_threshold = high_mean_threshold
        self.records: dict[str, PromptRecord] = {}
        self.pending_results: dict[str, list[PromptOutcome]] = {}

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
            grouped[self._group_record(record)].append(record)
        grouped[self.GROUP_1].sort(key=lambda record: (record.mean_reward, record.std_reward))
        grouped[self.GROUP_2].sort(key=lambda record: (-record.mean_reward, record.std_reward))
        grouped[self.GROUP_3].sort(key=lambda record: (-record.std_reward, record.mean_reward))
        return grouped

    def _update_record_with_results(self, results: list[tuple[str, list[PromptOutcome]]]) -> None:
        for prompt, outcomes in results:
            prompt = self._normalize_prompt(prompt)
            if prompt not in self.records:
                continue
            record = self.records[prompt]
            record.add_outcomes(outcomes)
            record.update_reward_stats()

    def add_new_prompts(self, prompts: list[str]) -> None:
        for prompt in prompts:
            prompt = self._normalize_prompt(prompt)
            self.records[prompt] = PromptRecord(prompt=prompt)
        while len(self.records) > self.max_prompts:
            self.records.pop(next(iter(self.records)))

    def add_prompt_outcome_to_record(self, prompt: str, prompt_outcome: PromptOutcome) -> None:
        self.pending_results.setdefault(self._normalize_prompt(prompt), []).append(prompt_outcome)

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

    def select_for_rollout(
        self,
        batch_size: int,
        step: int,
    ) -> list[str]:

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
        fill_index = 0
        while len(selected) < batch_size:
            selected.append(eligible[fill_index % len(eligible)])
            fill_index += 1
        for record in selected:
            record.last_selected_step = step
        return [record.prompt for record in selected]

    def synchronize(self, accelerator: Any | None) -> None:

        from accelerate.utils import gather_object

        gathered_results = gather_object(self.pending_results)
        self.pending_results.clear()
        if accelerator.is_main_process:
            valid_results = [
                result
                for result in gathered_results
                if isinstance(result, tuple) and len(result) == 2
            ]
            self._update_record_with_results(valid_results)


class DataGenerator:
    def __init__(
        self,
        topic: str,
        log_path: Path,
        model_name: str = "gpt-5.4-nano",
        history_size: int = 16,
        safe: bool = False,
        protected_data: list[str] | None = None,
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        filter_threshold: float = 0.75,
        oversample_factor: int = 4,
        max_attempts: int = 5,
        contamination_history_size: int = 512,
        new_prompts_per_step: int = 2,
    ) -> None:
        if safe and not protected_data:
            raise ValueError("DataGenerator requires protected_data when safe=True.")
        if new_prompts_per_step <= 0:
            raise ValueError("new_prompts_per_step must be positive.")
        if self.oversample_factor <= 0:
            raise ValueError("oversample_factor must be >= 1.")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be >= 1.")

        self.log_path = log_path
        self.history_size = history_size
        self.safe = safe
        self.data_filter = None
        self.oversample_factor = oversample_factor
        self.max_attempts = max_attempts
        self.new_prompts_per_step = new_prompts_per_step
        self.proposer = DataProposer(
            topic=topic,
            log_path=log_path,
            model_name=model_name,
        )
        self.contamination_buffer = None
        if safe:
            self.data_filter = DataFilter(
                data=protected_data or [],
                embedding_model_name=embedding_model_name,
                threshold=filter_threshold,
            )
            self.contamination_buffer = ContaminationPromptBuffer(max_history=contamination_history_size)

    def generate_prompts(
        self,
        generation_examples: dict[str, list[dict[str, Any]]],
        num_prompts: int,
        contamination_prompts: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.safe:
            return self.proposer.generate_prompts(
                generation_examples=generation_examples,
                num_prompts=num_prompts,
                contamination_prompts=contamination_prompts,
            )
        return self._generate_filtered_prompts(
            generation_examples=generation_examples,
            num_prompts=num_prompts,
            contamination_prompts=contamination_prompts,
        )

    def _generate_filtered_prompts(
        self,
        generation_examples: dict[str, list[dict[str, Any]]],
        num_prompts: int,
        contamination_prompts: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if num_prompts <= 0:
            raise ValueError("Number of prompts to generate must be positive.")

        accepted: list[dict[str, Any]] = []
        contaminated_history = list(contamination_prompts or [])
        for _ in range(self.max_attempts):
            num_candidates = max((num_prompts - len(accepted)) * self.oversample_factor, num_prompts)
            candidates = self.proposer.generate_prompts(
                generation_examples=generation_examples,
                num_prompts=num_candidates,
                contamination_prompts=contaminated_history,
            )
            candidate_prompts = [str(candidate["prompt"]).strip() for candidate in candidates]
            filter_result = self.data_filter.apply_filter(candidate_prompts)

            if self.contamination_buffer is not None:
                self.contamination_buffer.add_prompts(filter_result["contaminated"])
                contaminated_history = self.contamination_buffer.select_history()

            filtered_prompts = set(filter_result["filtered"])
            for candidate in candidates:
                if str(candidate["prompt"]).strip() in filtered_prompts:
                    accepted.append(candidate)
                if len(accepted) == num_prompts:
                    return accepted

        raise RuntimeError(f"DataGenerator produced {len(accepted)} filtered prompts, expected {num_prompts}.")

    def generate(
        self,
        *,
        buffer: PromptBuffer,
        num_prompts: int,
        step: int,
        accelerator: Any,
    ) -> list[str]:
        from accelerate.utils import broadcast_object_list

        if num_prompts <= 0:
            return []

        buffer.synchronize(accelerator)
        process_index = int(getattr(accelerator, "process_index", 0))

        if self.safe:
            self.contamination_buffer.synchronize(accelerator)

        payload: list[Any] = [None]
        if accelerator.is_main_process:
            generation_examples = buffer.select_for_generation(k=self.history_size)
            num_new_prompts = max(
                self.new_prompts_per_step,
                num_prompts - len(buffer.records),
            )
            generated = self.generate_prompts(
                generation_examples=generation_examples,
                num_prompts=num_new_prompts,
                contamination_prompts=(
                    self.contamination_buffer.select_history()
                    if self.safe
                    else None
                ),
            )
            buffer.add_prompts(
                [str(candidate["prompt"]).strip() for candidate in generated],
            )
            selected = buffer.select_for_rollout(
                batch_size=num_prompts,
                step=step,
            )
            payload[0] = {
                "prompts": selected,
                "generation_examples": {
                    group: len(items) for group, items in generation_examples.items()
                },
                "prompt_pool_size": len(buffer.records),
            }
        broadcast_object_list(payload, from_process=0)

        if self.safe:
            self.contamination_buffer.synchronize(accelerator)

        batch_info = payload[0] if isinstance(payload[0], dict) else {}
        selected = batch_info.get("prompts", [])
        if len(selected) != num_prompts:
            raise RuntimeError(
                f"DataGenerator selected {len(selected)} prompts, expected {num_prompts}."
            )
        final_prompts = [str(prompt).strip() for prompt in selected]
        log_event(
            self.log_path,
            {
                "type": "data_generator_batch",
                "step": step,
                "batch_size": len(final_prompts),
                "generation_examples": batch_info.get("generation_examples", {}),
                "prompt_pool_size": batch_info.get("prompt_pool_size", 0),
                "num_selected_prompts": num_prompts,
                "process_index": process_index,
                "selected_prompts": final_prompts,
            },
        )
        return final_prompts
