from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .data_filter import ContaminationPromptBuffer, DataFilter
from .data_proposer import DataProposer
from src.logging import log_event
from .prompt_buffer import PromptBuffer
from src.data_generator.data_proposer import GeneratorMode


class DataGenerator:
    def __init__(
        self,
        log_path: Path,
        # proposer configuration
        topic: str,
        model_name: str = "gpt-5.4-nano",
        mode: GeneratorMode = "natural",
        # filtering configuration
        protected_data: list[str] | None = None,
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        filter_threshold: float = 0.75,
        contamination_history_size: int = 512,
        # final generation configuration
        safe: bool = False,
        generation_context_size: int = 16,
        new_prompts_per_step: float = 0.5,
        oversample_factor: int = 4,
        max_attempts: int = 5,
    ) -> None:
        if safe and not protected_data:
            raise ValueError("DataGenerator requires protected_data when safe=True.")
        self._validate_new_prompts_per_step(new_prompts_per_step)
        if oversample_factor <= 0:
            raise ValueError("oversample_factor must be >= 1.")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be >= 1.")

        self.log_path = log_path
        # generation configuration
        self.generation_context_size = generation_context_size
        self.safe = safe
        self.oversample_factor = oversample_factor
        self.max_attempts = max_attempts
        self.new_prompts_per_step = float(new_prompts_per_step)
        # proposer configuration
        self.mode = mode
        self.topic = topic
        self.model_name = model_name
        self.proposer = DataProposer(
            topic=topic,
            log_path=log_path,
            model_name=model_name,
            mode=mode
        )
        # filtering configuration
        self.contamination_buffer = None
        self.data_filter = None
        self.filter_threshold = filter_threshold
        self.embedding_model_name = embedding_model_name
        if safe:
            self.data_filter = DataFilter(
                data=protected_data or [],
                embedding_model_name=embedding_model_name,
                threshold=filter_threshold,
            )
            self.contamination_buffer = ContaminationPromptBuffer(max_history=contamination_history_size)

    def generate_prompts(
        self,
        rollout_group_context: dict[str, list[dict[str, Any]]],
        num_prompts: int,
        contamination_prompts: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.safe:
            return self.proposer.generate_prompts(
                rollout_group_context=rollout_group_context,
                num_prompts=num_prompts,
                contamination_prompts=contamination_prompts,
            )
        return self._generate_filtered_prompts(
            rollout_group_context=rollout_group_context,
            num_prompts=num_prompts,
            contamination_prompts=contamination_prompts,
        )

    def _generate_filtered_prompts(
        self,
        rollout_group_context: dict[str, list[dict[str, Any]]],
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
                rollout_group_context=rollout_group_context,
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
            rollout_group_context = buffer.select_for_generation(
                max_context_prompts=self.generation_context_size
            )
            prompt_pool_size_before_generation = len(buffer.records)
            num_new_prompts = self._resolve_num_new_prompts(
                batch_size=num_prompts,
                prompt_pool_size=prompt_pool_size_before_generation,
            )
            generated = self.generate_prompts(
                rollout_group_context=rollout_group_context,
                num_prompts=num_new_prompts,
                contamination_prompts=(
                    self.contamination_buffer.select_history()
                    if self.safe
                    else None
                ),
            )
            buffer.add_generated_prompts(
                [str(candidate["prompt"]).strip() for candidate in generated],
            )
            selected = buffer.select_for_rollout(
                batch_size=num_prompts,
                step=step,
            )
            payload[0] = {
                "prompts": selected,
                "rollout_group_context": {
                    group: len(items) for group, items in rollout_group_context.items()
                },
                "prompt_pool_size": len(buffer.records),
                "num_new_prompts": num_new_prompts,
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
                "rollout_group_context": batch_info.get("rollout_group_context", {}),
                "prompt_pool_size": batch_info.get("prompt_pool_size", 0),
                "num_selected_prompts": num_prompts,
                "num_new_prompts_per_step": batch_info.get(
                    "num_new_prompts",
                    self._resolve_new_prompts_per_step(num_prompts),
                ),
                "process_index": process_index,
                "selected_prompts": final_prompts,
            },
        )
        return final_prompts

    @staticmethod
    def _validate_new_prompts_per_step(value: float) -> None:
        if not 0 < float(value) <= 1:
            raise ValueError("new_prompts_per_step must be a float in (0, 1].")

    def _resolve_new_prompts_per_step(self, batch_size: int) -> int:
        return max(1, math.ceil(batch_size * float(self.new_prompts_per_step)))

    def _resolve_num_new_prompts(self, batch_size: int, prompt_pool_size: int) -> int:
        if prompt_pool_size < batch_size:
            return batch_size
        return self._resolve_new_prompts_per_step(batch_size)
