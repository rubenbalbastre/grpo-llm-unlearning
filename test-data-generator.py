import json
import os
from pathlib import Path

from dotenv import load_dotenv
import weave

from src.data_generator import CompletionRecord, DataGenerator


def main():

    # logging
    load_dotenv()
    weave.init(os.environ["WANDB_PROJECT"])
    
    # topic
    forget_concept = "Stephen King"
    history = [
        CompletionRecord(
            prompt="Who is Stephen King?",
            completion="Stephen King is an American author.",
            reward=0.0,
        ),
        CompletionRecord(
            prompt="Name a famous book written by Stephen King.",
            completion="I cannot provide that information.",
            reward=1.0,
        ),
    ]

    # generate data
    output_path = Path("outputs/data_generator_events.jsonl")
    data_generator = DataGenerator(topic=f"Forget concept: '{forget_concept}.'", log_path=output_path)
    prompts = data_generator.generate_prompts(
        history=history,
        n=4
    )

    # write the new prompts to a file
    Path("outputs").mkdir(parents=True, exist_ok=True)
    with open("outputs/data_generator_prompts.json", "w") as f:
        json.dump(prompts, f, indent=4)
    
    print(prompts)


if __name__ == "__main__":

    main()
