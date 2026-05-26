from __future__ import annotations

import re
from pathlib import Path

from src.logging import log_event
from src.prompt_buffer import PromptBuffer, RolloutCompletionOutcome


REFUSAL_PATTERNS = [
    "Stephen King",
    "Carrie",
    "The Shining",
    "It",
    "Misery",
    "Pet Sematary",
    "The Stand",
    "The Dark Tower",
    "The Green Mile",
    "Christine",
    "Cujo",
    "Salem's Lot",
    "Doctor Sleep",
    "Under the Dome",
    "11/22/63",
    "The Dead Zone",
    "Firestarter",
    "Needful Things",
    "Gerald's Game",
    "Bag of Bones",
    "Insomnia",
    "The Mist",
    "Rita Hayworth and Shawshank Redemption",
    "Hearts in Atlantis",
    "The Long Walk",
    "Cycle of the Werewolf",
    "Night Shift",
    "Skeleton Crew",
    "The Girl Who Loved Tom Gordon",
    "Revival",
    "The Institute",
    "Pennywise the Dancing Clown",
    "Jack Torrance",
    "Danny Torrance",
    "Wendy Torrance",
    "Carrie White",
    "Annie Wilkes",
    "Roland Deschain",
    "Randall Flagg",
    "Overlook Hotel",
    "Derry",
    "Castle Rock",
    "Losers' Club",
    "Blaine the Mono",
    "Ka-tet",
    "Alan Pangborn",
    "Billy Summers",
    "Mr. Mercedes",
    "Holly Gibney",
    "Brady Hartsfield",
    "The Outsider",
    "If It Bleeds",
    "End of Watch",
    "Finders Keepers",
    "Joyland",
    "Duma Key",
    "Lisey's Story",
    "The Colorado Kid",
    "From a Buick 8",
    "Storm of the Century",
    "Desperation"
]

ENTITY_MATCHERS = [
    (
        entity,
        re.compile(rf"(?<!\w){re.escape(entity)}(?!\w)", flags=re.IGNORECASE),
    )
    for entity in REFUSAL_PATTERNS
]


def matched_entities(text: str) -> list[str]:
    return [entity for entity, matcher in ENTITY_MATCHERS if matcher.search(text)]


def entity_forgetting_reward(text: str) -> tuple[float, list[str]]:
    entities = matched_entities(text)
    return 1.0 / (1.0 + len(entities)), entities


def binary_forgetting_reward(text: str) -> tuple[float, list[str]]:
    entities = matched_entities(text)
    return (1.0 if not entities else 0.0), entities


def make_unlearning_reward_func(
    buffer: PromptBuffer | None,
    log_path: Path,
    reward_mode: str = "entity_count",
):
    reward_functions = {
        "binary": binary_forgetting_reward,
        "entity_count": entity_forgetting_reward,
    }
    if reward_mode not in reward_functions:
        raise ValueError(
            f"reward.mode must be one of {list(reward_functions)}, got {reward_mode!r}."
        )
    compute_reward = reward_functions[reward_mode]

    def unlearning_reward_func(prompts, completions, **kwargs) -> list[float]:
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

        for p, c in zip(prompts_list, completions_list, strict=True):
            reward, entities = compute_reward(str(c))

            if buffer is not None:
                buffer.record_rollout_outcome(
                    str(p),
                    RolloutCompletionOutcome(completion=str(c), reward=reward, step=step),
                )
            rewards.append(reward)
            log_event(
                log_path,
                {
                    "type": "reward",
                    "prompt": str(p),
                    "completion": str(c),
                    "reward": reward,
                    "reward_mode": reward_mode,
                    "matched_entities": entities,
                    "matched_entity_count": len(entities),
                    "step": step,
                },
            )

        return rewards

    return unlearning_reward_func
