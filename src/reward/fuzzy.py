import re
import unicodedata
from dataclasses import dataclass
from rapidfuzz import fuzz
from pathlib import Path
from typing import Iterable

from src.data_generator.prompt_buffer import PromptBuffer, RolloutCompletionOutcome
from src.reward.refusal_patterns import REFUSAL_PATTERNS

EntityLike = str | tuple[str, re.Pattern]
FUZZY_SIMILARITY_THRESHOLD = 85.0
MIN_FUZZY_ENTITY_LENGTH = 4


@dataclass(frozen=True)
class FuzzyEntity:
    name: str
    normalized: str
    compact: str
    normalized_pattern: re.Pattern


def normalize(text: str) -> str:
    text = text.casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_fuzzy_entities(forbidden_entities: Iterable[EntityLike]) -> list[FuzzyEntity]:
    entities: list[FuzzyEntity] = []
    for entity in iter_entity_names(forbidden_entities):
        normalized = normalize(entity)
        if not normalized:
            continue
        entities.append(
            FuzzyEntity(
                name=entity,
                normalized=normalized,
                compact=normalized.replace(" ", ""),
                normalized_pattern=re.compile(rf"(?<!\w){re.escape(normalized)}(?!\w)"),
            )
        )
    return entities


def build_fuzzy_entities_for_concept(forget_concept: str) -> list[FuzzyEntity]:
    if forget_concept not in REFUSAL_PATTERNS:
        raise ValueError(
            "No refusal patterns configured for "
            f"{forget_concept!r}. Available options: {list(REFUSAL_PATTERNS)}."
        )
    return build_fuzzy_entities(REFUSAL_PATTERNS[forget_concept])


def has_normalized_match(normalized_output: str, entity: FuzzyEntity) -> bool:
    return entity.normalized_pattern.search(normalized_output) is not None


def has_fuzzy_match(
    compact_output: str,
    entity: FuzzyEntity,
    similarity_threshold: float = FUZZY_SIMILARITY_THRESHOLD,
) -> bool:
    if (
        not compact_output
        or not entity.compact
        or len(entity.compact) < MIN_FUZZY_ENTITY_LENGTH
    ):
        return False

    return (
        fuzz.partial_ratio(
            entity.compact,
            compact_output,
            score_cutoff=similarity_threshold,
        )
        >= similarity_threshold
    )


def make_forgetting_fuzzy_reward_func(
    buffer: PromptBuffer | None,
    log_path: Path,
    forget_concept: str,
    reward_mode: str = "entity_count",
    log_events: bool = True,
):

    entity_matchers = build_fuzzy_entities_for_concept(forget_concept)

    def forgetting_fuzzy_reward(prompts, completions, **kwargs) -> list[float]:
        rewards: list[float] = []
        trainer_state = kwargs.get("trainer_state")
        step = getattr(trainer_state, "global_step", None)

        selected_prompts = kwargs.get("selected_prompt", prompts)
        prompts_list = list(selected_prompts) if selected_prompts is not None else []
        completions_list = list(completions) if completions is not None else []
        if len(prompts_list) != len(completions_list):
            raise ValueError(
                f"Reward input mismatch: {len(prompts_list)} prompts, "
                f"{len(completions_list)} completions."
            )
        if not prompts_list:
            return rewards

        for prompt, completion in zip(prompts_list, completions_list, strict=True):
            reward = compute_fuzzy_reward_per_completion(str(completion), entity_matchers)

            if buffer is not None:
                buffer.record_rollout_outcome(
                    str(prompt),
                    RolloutCompletionOutcome(
                        completion=str(completion),
                        reward=reward,
                        step=step,
                    ),
                )
            rewards.append(reward)

        return rewards

    return forgetting_fuzzy_reward



def iter_entity_names(forbidden_entities: Iterable[EntityLike]) -> Iterable[str]:
    for entity in forbidden_entities:
        if isinstance(entity, tuple):
            yield entity[0]
        else:
            yield entity


def compute_fuzzy_reward_per_completion(
    completion: str,
    forbidden_entities: Iterable[EntityLike] | Iterable[FuzzyEntity],
) -> float:
    raw_entities = list(forbidden_entities)
    entities = (
        raw_entities
        if all(isinstance(entity, FuzzyEntity) for entity in raw_entities)
        else build_fuzzy_entities(raw_entities)
    )
    normalized_completion = normalize(completion)
    compact_completion = normalized_completion.replace(" ", "")

    for entity in entities:
        if has_normalized_match(normalized_completion, entity):
            return 0.0
        if has_fuzzy_match(compact_completion, entity):
            return 0.0

    return 1.0
