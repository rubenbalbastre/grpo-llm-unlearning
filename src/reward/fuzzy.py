import re
import unicodedata
from rapidfuzz.distance import Levenshtein
from pathlib import Path

from src.data_generator.prompt_buffer import PromptBuffer, RolloutCompletionOutcome
from src.reward.forgetting import build_entity_matchers


def normalize(text: str) -> str:
    text = text.casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def max_edit_distance(entity: str, ratio: float = 0.20) -> int:
    n = len(entity.replace(" ", ""))
    return max(1, round(n * ratio))


def has_normalized_match(output: str, entity: str) -> bool:
    output = normalize(output)
    entity = normalize(entity)

    return re.search(rf"(?<!\w){re.escape(entity)}(?!\w)", output) is not None


def has_fuzzy_match(
    output: str,
    entity: str,
    similarity_threshold: float = 0.85,
) -> bool:
    output = normalize(output).replace(" ", "")
    entity = normalize(entity).replace(" ", "")

    if not output or not entity:
        return False

    max_dist = max_edit_distance(entity)
    entity_len = len(entity)

    for i in range(len(output)):
        for j in range(i + max(1, entity_len - max_dist), min(len(output), i + entity_len + max_dist) + 1):
            span = output[i:j]

            sim = Levenshtein.normalized_similarity(entity, span)
            dist = Levenshtein.distance(entity, span)

            if sim >= similarity_threshold and dist <= max_dist:
                return True

    return False


def make_forgetting_fuzzy_reward_func(
    buffer: PromptBuffer | None,
    log_path: Path,
    forget_concept: str,
    reward_mode: str = "entity_count",
    log_events: bool = True,
):

    entity_matchers = build_entity_matchers(forget_concept)

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



def compute_fuzzy_reward_per_completion(completion: str, forbidden_entities: list[str]) -> float:
    for entity in forbidden_entities:
        if has_normalized_match(completion, entity):
            return 0.0
        if has_fuzzy_match(completion, entity):
            return 0.0

    return 1.0
