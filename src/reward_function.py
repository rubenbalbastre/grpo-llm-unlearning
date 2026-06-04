from __future__ import annotations

import re
from pathlib import Path

from src.logging import log_event
from src.data_generator.prompt_buffer import PromptBuffer, RolloutCompletionOutcome


REFUSAL_PATTERNS = {
    "Stephen King": [
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
    ],
    "Confucius": [
        "Confucius",
        "Confucianism",
        "Kongzi",
        "Kong Fuzi",
        "Master Kong",
        "Analects",
        "Ren",
        "Yi",
        "Li",
        "Zhi",
        "Xin",
        "Xiao",
        "Filial Piety",
        "Junzi",
        "Five Relationships",
        "Wulun",
        "Five Constants",
        "Wuchang",
        "Ritual Propriety",
        "Benevolent Rulership",
        "Civil Service Examination",
        "Imperial Examination System",
        "Scholar-Official",
        "Mandarinate",
        "Taixue",
        "Spring and Autumn Period",
        "Eastern Zhou",
        "Great Learning",
        "Doctrine of the Mean",
        "Book of Rites",
        "Classic of Filial Piety",
        "Four Books",
        "Five Classics",
        "Neo-Confucianism",
        "Cheng-Zhu School",
        "Qufu",
        "Mount Ni",
        "State of Lu",
        "Benevolent Governance",
        "Hierarchical Social Order",
        "Ruler–Subject Bond",
        "Father–Son Bond",
        "Husband–Wife Bond",
        "Elder Brother–Younger Brother Bond",
        "Friend–Friend Bond",
        "Scholar-Gentry Class",
        "Imperial Academy",
        "Duke of Zhou",
        "Moral Rectification",
        "Virtuous Ruler",
        "Mencius",
        "Xunzi",
        "Han Learning",
        "Imperial Confucian Canon",
        "Temple of Confucius",
        "Confucius Shrine",
        "Seokjeonjeon",
        "Sungkyunkwan",
        "Sishu",
        "Shujing"
    ]
}

def build_entity_matchers(forget_concept: str) -> list[tuple[str, re.Pattern]]:
    if forget_concept not in REFUSAL_PATTERNS:
        raise ValueError(
            "No refusal patterns configured for "
            f"{forget_concept!r}. Available options: {list(REFUSAL_PATTERNS)}."
        )
    entities = REFUSAL_PATTERNS[forget_concept]
    return [
        (
            entity,
            re.compile(rf"(?<!\w){re.escape(entity)}(?!\w)", flags=re.IGNORECASE),
        )
        for entity in entities
    ]


def matched_entities(text: str, matchers: list[tuple[str, re.Pattern]]) -> list[str]:
    return [entity for entity, matcher in matchers if matcher.search(text)]


def entity_forgetting_reward(
    text: str,
    matchers: list[tuple[str, re.Pattern]],
) -> tuple[float, list[str]]:
    entities = matched_entities(text, matchers)
    return 1.0 / (1.0 + len(entities)), entities


def binary_forgetting_reward(
    text: str,
    matchers: list[tuple[str, re.Pattern]],
) -> tuple[float, list[str]]:
    entities = matched_entities(text, matchers)
    return (1.0 if not entities else 0.0), entities


def make_unlearning_reward_func(
    buffer: PromptBuffer | None,
    log_path: Path,
    forget_concept: str,
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
    entity_matchers = build_entity_matchers(forget_concept)

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
            reward, entities = compute_reward(str(c), entity_matchers)

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
                    "forget_concept": forget_concept,
                    "matched_entities": entities,
                    "matched_entity_count": len(entities),
                    "step": step,
                },
            )

        return rewards

    return unlearning_reward_func
