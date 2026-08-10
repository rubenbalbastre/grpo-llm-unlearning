import math
import re
import unicodedata
from dataclasses import dataclass
from rapidfuzz import fuzz
from typing import Any, Iterable

from src.reward.components.constants.target_patterns import get_target_patterns
from src.reward.components.simple_match import count_words

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


def build_fuzzy_entities_for_concept(
    forget_concept: str,
    pattern_splits: str | Iterable[str] = ("hard", "soft"),
) -> list[FuzzyEntity]:
    return build_fuzzy_entities(get_target_patterns(forget_concept, pattern_splits))


def has_normalized_match(normalized_output: str, entity: FuzzyEntity) -> bool:
    return entity.normalized_pattern.search(normalized_output) is not None


def normalized_matches(normalized_output: str, entity: FuzzyEntity) -> list[str]:
    return [entity.name for _ in entity.normalized_pattern.finditer(normalized_output)]


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
    forget_concept: str,
    reward_mode: str = "entity_count",
    pattern_splits: str | Iterable[str] = ("hard", "soft"),
    log_prefix: str = "fuzzy",
    length_aware_min_words: int = 50,
    length_aware_alpha: float = 20.0,
):
    if length_aware_min_words <= 0:
        raise ValueError(
            "reward.functions.fuzzy_match.length_aware_min_words must be >= 1."
        )
    if length_aware_alpha < 0:
        raise ValueError(
            "reward.functions.fuzzy_match.length_aware_alpha must be >= 0."
        )
    reward_modes = {"binary", "length_aware"}
    if reward_mode not in reward_modes:
        raise ValueError(
            f"reward.functions.fuzzy_match.mode must be one of {sorted(reward_modes)}, "
            f"got {reward_mode!r}."
        )

    entity_matchers = build_fuzzy_entities_for_concept(forget_concept, pattern_splits)

    def forgetting_fuzzy_reward(prompts, completions, **kwargs) -> list[float]:
        rewards: list[float] = []
        log_extra = kwargs.get("log_extra")

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

        matched_entities_log: list[str] = []
        matched_entity_count_log: list[int] = []
        word_count_log: list[int] = []
        for completion in completions_list:
            completion_text = str(completion)
            if reward_mode == "length_aware":
                reward, matched_entities = compute_length_aware_fuzzy_reward(
                    completion_text,
                    entity_matchers,
                    min_words=length_aware_min_words,
                    alpha=length_aware_alpha,
                )
            else:
                reward, matched_entities = compute_fuzzy_reward_per_completion(
                    completion_text,
                    entity_matchers,
                )

            rewards.append(reward)
            matched_entities_log.append("|".join(matched_entities))
            matched_entity_count_log.append(len(matched_entities))
            word_count_log.append(count_words(completion_text))

        if log_extra is not None:
            log_extra(f"{log_prefix}_reward", rewards)
            log_extra(f"{log_prefix}_entities", matched_entities_log)
            log_extra(f"{log_prefix}_entity_count", matched_entity_count_log)
            log_extra(f"{log_prefix}_word_count", word_count_log)

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
) -> tuple[float, list[str]]:
    raw_entities = list(forbidden_entities)
    entities = (
        raw_entities
        if all(isinstance(entity, FuzzyEntity) for entity in raw_entities)
        else build_fuzzy_entities(raw_entities)
    )
    normalized_completion = normalize(completion)
    compact_completion = normalized_completion.replace(" ", "")

    matched_entities: list[str] = []
    for entity in entities:
        if has_normalized_match(normalized_completion, entity):
            matched_entities.append(entity.name)
            continue
        if has_fuzzy_match(compact_completion, entity):
            matched_entities.append(entity.name)

    return (1.0 if not matched_entities else 0.0), matched_entities


def compute_fuzzy_matches(
    completion: str,
    forbidden_entities: Iterable[EntityLike] | Iterable[FuzzyEntity],
) -> list[str]:
    raw_entities = list(forbidden_entities)
    entities = (
        raw_entities
        if all(isinstance(entity, FuzzyEntity) for entity in raw_entities)
        else build_fuzzy_entities(raw_entities)
    )
    normalized_completion = normalize(completion)
    compact_completion = normalized_completion.replace(" ", "")

    matched_entities: list[str] = []
    for entity in entities:
        exact_matches = normalized_matches(normalized_completion, entity)
        if exact_matches:
            matched_entities.extend(exact_matches)
        elif has_fuzzy_match(compact_completion, entity):
            matched_entities.append(entity.name)
    return matched_entities


def compute_length_aware_fuzzy_reward(
    completion: str,
    forbidden_entities: Iterable[EntityLike] | Iterable[FuzzyEntity],
    min_words: int,
    alpha: float,
) -> tuple[float, list[str]]:
    matched_entities = compute_fuzzy_matches(completion, forbidden_entities)
    effective_words = max(count_words(completion), min_words)
    match_density = len(matched_entities) / effective_words
    return math.exp(-alpha * match_density), matched_entities


def build_fuzzy_match_reward(
    config: Any,
    *,
    forget_concept: str,
):
    config = config or {}
    return make_forgetting_fuzzy_reward_func(
        forget_concept=forget_concept,
        reward_mode=config.get("mode", "binary"),
        pattern_splits=config.get("pattern_splits", ["hard", "soft"]),
        log_prefix=config.get("log_prefix", "fuzzy"),
        length_aware_min_words=int(config.get("length_aware_min_words", 50)),
        length_aware_alpha=float(config.get("length_aware_alpha", 20.0)),
    )
