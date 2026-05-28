import json
import os
from pathlib import Path

from dotenv import load_dotenv
import weave

from src.data_generator.data_generator import DataGenerator
from src.data_generator.prompt_buffer import PromptBuffer, RolloutCompletionOutcome


def main():

    # logging
    load_dotenv()
    weave.init(os.environ["WANDB_PROJECT"])

    # topic
    forget_concept = "Stephen King"
    buffer = PromptBuffer()
    prompts_and_outcomes = {
        "Who is Stephen King?": [("Stephen King is an American author.", 0.0)],
        "Name a famous book written by Stephen King.": [("I cannot provide that information.", 1.0)],
        "Discuss the author of The Shining.": [
            ("Stephen King wrote it.", 0.0),
            ("I cannot provide that information.", 1.0),
        ],
    }
    buffer.add_generated_prompts(list(prompts_and_outcomes))
    for prompt, outcomes in prompts_and_outcomes.items():
        for completion, reward in outcomes:
            buffer.record_rollout_outcome(
                prompt,
                RolloutCompletionOutcome(completion=completion, reward=reward, step=0),
            )
    buffer.synchronize(None)
    rollout_group_context = buffer.select_for_generation(max_context_prompts=3)

    # # generate safe data
    # output_path = Path("outputs/safe_data_generator_events.jsonl")
    # data_generator = DataGenerator(
    #     topic=f"Forget concept: '{forget_concept}.'",
    #     log_path=output_path,
    #     safe=True,
    #     filter_threshold=0.55,
    #     protected_data=[
    #         "Stephen King is an American author of horror, supernatural fiction, suspense, and fantasy novels.",
    #         "Carrie, The Shining, It, Misery, and The Stand are works associated with Stephen King.",
    #     ],
    # )
    # prompts = data_generator.generate_prompts(
    #     rollout_group_context=rollout_group_context,
    #     num_prompts=4,
    # )

    # # write the new prompts to a file
    # Path("outputs").mkdir(parents=True, exist_ok=True)
    # with open("outputs/safe_data_generator_prompts.json", "w") as f:
    #     json.dump(prompts, f, indent=4)

    # print(prompts)

    # generate data
    output_path = Path("outputs/data_generator_events.jsonl")
    data_generator = DataGenerator(
        topic=f"Forget concept: '{forget_concept}.'",
        log_path=output_path,
        safe=False,
        mode="expanded"
    )
    prompts = data_generator.generate_prompts(
        rollout_group_context=rollout_group_context,
        num_prompts=4,
    )

    # write the new prompts to a file
    Path("outputs").mkdir(parents=True, exist_ok=True)
    with open("outputs/data_generator_prompts.json", "w") as f:
        json.dump(prompts, f, indent=4)

    print(prompts)

if __name__ == "__main__":

    main()
