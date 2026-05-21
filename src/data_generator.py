from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data_filter import ContaminationPromptBuffer, DataFilter
from src.data_proposer import DataProposer
from src.logging import log_event


@dataclass
class CompletionRecord:
    prompt: str
    completion: str
    reward: float
    step: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AdaptivePromptBuffer:
    def __init__(self, max_history: int = 512) -> None:
        self.max_history = max_history
        self.records: list[CompletionRecord] = []

    def add_record(self, record: CompletionRecord) -> None:
        self.records.append(record)
        if len(self.records) > self.max_history:
            self.records = self.records[-self.max_history :]

    def select_history(self, k: int = 16) -> list[CompletionRecord]:
        return self.records[-k:]

    def synchronize(self, accelerator: Any | None) -> None:
        if accelerator is None:
            return

        from accelerate.utils import broadcast_object_list, gather_object

        gathered_records = gather_object(self.records)
        payload: list[Any] = [None]
        if accelerator.is_main_process:
            seen: set[tuple[str, str, float, int | None]] = set()
            merged_records: list[CompletionRecord] = []
            for record in gathered_records:
                if not isinstance(record, CompletionRecord):
                    continue
                key = (record.prompt, record.completion, record.reward, record.step)
                if key not in seen:
                    seen.add(key)
                    merged_records.append(record)
            payload[0] = merged_records[-self.max_history :]
        broadcast_object_list(payload, from_process=0)
        if isinstance(payload[0], list):
            self.records = payload[0]


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
    ) -> None:
        if safe and not protected_data:
            raise ValueError("DataGenerator requires protected_data when safe=True.")

        self.log_path = log_path
        self.history_size = history_size
        self.safe = safe
        self.data_filter = None
        self.oversample_factor = oversample_factor
        self.max_attempts = max_attempts
        self.proposer = DataProposer(
            topic=topic,
            log_path=log_path,
            model_name=model_name,
            history_size=history_size,
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
        history: list[CompletionRecord],
        num_prompts: int,
        contamination_prompts: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.safe:
            return self.proposer.generate_prompts(
                history=history,
                num_prompts=num_prompts,
                contamination_prompts=contamination_prompts,
            )
        return self._generate_filtered_prompts(
            history=history,
            num_prompts=num_prompts,
            contamination_prompts=contamination_prompts,
        )

    def _generate_filtered_prompts(
        self,
        history: list[CompletionRecord],
        num_prompts: int,
        contamination_prompts: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if num_prompts <= 0:
            raise ValueError("Number of prompts to generate must be positive.")
        if self.oversample_factor <= 0:
            raise ValueError("oversample_factor must be >= 1.")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be >= 1.")
        if self.data_filter is None:
            raise RuntimeError("DataGenerator safe mode requires data_filter.")

        accepted: list[dict[str, Any]] = []
        contaminated_history = list(contamination_prompts or [])
        for _ in range(self.max_attempts):
            num_candidates = max((num_prompts - len(accepted)) * self.oversample_factor, num_prompts)
            candidates = self.proposer.generate_prompts(
                history=history,
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
        buffer: AdaptivePromptBuffer,
        batch_size: int,
        step: int,
        accelerator: Any,
    ) -> list[str]:
        from accelerate.utils import broadcast_object_list

        if batch_size <= 0:
            return []

        buffer.synchronize(accelerator)
        history = buffer.select_history(k=self.history_size)
        num_processes = int(getattr(accelerator, "num_processes", 1))
        process_index = int(getattr(accelerator, "process_index", 0))
        num_prompts = batch_size * num_processes

        if self.safe:
            self.contamination_buffer.synchronize(accelerator)

        payload: list[Any] = [None]
        if accelerator.is_main_process:
            payload[0] = self.generate_prompts(
                history=history,
                num_prompts=num_prompts,
                contamination_prompts=(
                    self.contamination_buffer.select_history()
                    if self.safe
                    else None
                ),
            )
        broadcast_object_list(payload, from_process=0)

        if self.safe:
            self.contamination_buffer.synchronize(accelerator)

        generated = payload[0] if isinstance(payload[0], list) else []

        start = process_index * batch_size
        end = start + batch_size
        selected = generated[start:end]
        if len(selected) < batch_size:
            raise RuntimeError(
                f"DataGenerator returned {len(generated)} prompts, but process {process_index} needs slice "
                f"[{start}:{end}]. "
                "Increase the generated prompt count."
            )
        final_prompts = [str(candidate["prompt"]).strip() for candidate in selected]
        log_event(
            self.log_path,
            {
                "type": "data_generator_batch",
                "step": step,
                "batch_size": len(final_prompts),
                "history_size": len(history),
                "num_generated_prompts": len(generated),
                "process_index": process_index,
                "final_prompts": final_prompts,
            },
        )
        return final_prompts
