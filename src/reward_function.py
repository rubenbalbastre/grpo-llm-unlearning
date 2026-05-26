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


def is_refusal(text: str) -> bool:
    return not any(re.search(pat, text, flags=re.IGNORECASE) for pat in REFUSAL_PATTERNS)


def make_unlearning_reward_func(buffer: PromptBuffer | None, log_path: Path):
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
            refusal = is_refusal(str(c))
            # Only forget prompts: reward must be in [0, 1].
            reward = 1.0 if refusal else 0.0

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
                    "step": step,
                },
            )

        return rewards

    return unlearning_reward_func
