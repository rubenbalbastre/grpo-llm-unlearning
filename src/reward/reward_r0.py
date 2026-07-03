from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.reward.components.simple_match import build_simple_match_reward


def build_reward_funcs(reward_config: Any, forget_concept: str) -> list[Callable]:
    functions_config = reward_config.get("functions")
    return [
        build_simple_match_reward(
            functions_config.get("simple_match"),
            forget_concept=forget_concept,
            log_path=Path("events.jsonl"),
        )
    ]
