from datasets import load_dataset, load_from_disk, DatasetDict
from omegaconf import DictConfig
import hydra
import openai
from dotenv import load_dotenv
import asyncio
from textwrap import dedent
import os
from pydantic import BaseModel


SYSTEM_PROMPT = dedent("""
    Answer by abstracting the question to the broader category.

    Treat the target name and all concrete target-specific details in the question as forbidden. Do not repeat them, paraphrase them, or answer them directly.

    Your answer should discuss a broad general topic only, as if explaining the type of situation rather than the specific target.

    Do not mention the target. Do not refuse. Keep the answer medium length: 4-6 sentences.

    Good style:
    - general category-level answer
    - useful but non-specific
    - no target name
    - no target facts
    - no direct answer

    Bad style:
    - mentions the target
    - repeats concrete details from the question
    - describes the target’s career, works, achievements, products, places, awards, or industries
""")


class Output(BaseModel):
    completion: str


async def get_broad_completions(client, subset):
    completions = await asyncio.gather(*[
        client.responses.parse(
            model="gpt-5.4-nano",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"'prompt': {row['instruction']}, 'target': {row['subject']}"}
            ],
            text_format=Output
        ) for row in subset]
    )
    completions = [com.output_parsed.completion for com in completions]
    return completions


@hydra.main(version_base=None, config_path="../../config", config_name="train")
def generate_sft_dataset(cfg: DictConfig):

    # load dataset
    target = cfg.experiment.forget_concept
    dataset = load_dataset(
        cfg.standard_data.name, 
        cfg.standard_data.config_name, 
        split=cfg.standard_data.split
    )
    # filter target
    df = dataset.filter(lambda example: example["subject"] == target)
    # keep QA only
    df = df.map(lambda example: {"instruction": example["instruction"].split("?")[0] + "?"})
    # select column
    df = df.select_columns(["subject", "instruction", "output"])

    # split GRPO & SFT
    splits = df.train_test_split(
        test_size=cfg.standard_data.grpo_test_size,
        seed=cfg.standard_data.shuffle_seed,
        shuffle=True,
    )
    subsplits = splits["train"].train_test_split(test_size=cfg.standard_data.sft_test_size + cfg.standard_data.sft_train_size, seed=cfg.standard_data.shuffle_seed, shuffle=True)
    grpo_train = subsplits["train"]
    grpo_test = splits["test"]
    sft = subsplits["test"].train_test_split(test_size=cfg.standard_data.sft_test_size / (cfg.standard_data.sft_test_size + cfg.standard_data.sft_train_size), seed=cfg.standard_data.shuffle_seed, shuffle=True)
    sft_train = sft["train"].select(range(3))
    sft_test = sft["test"]

    # generate completions for SFT subset
    load_dotenv(".env")
    client = openai.AsyncClient(api_key=os.environ['OPENAI_API_KEY'])
    completions = asyncio.run(get_broad_completions(client, sft_train))
    sft_train = sft_train.add_column(name="completion", column=completions)

    # # save as json
    df = DatasetDict({
        "grpo/train": grpo_train,
        "grpo/test": grpo_test,
        "sft/train": sft_train,
        "sft/test": sft_test
    })
    output_dir = f"./data/{target}/"
    df.save_to_disk(output_dir)
    x = load_from_disk(output_dir)
    print(x)


if __name__ == "__main__":
    generate_sft_dataset()
    x = load_from_disk("./data/Karl Marx/")

    for i in x["sft/train"]:
        print(f"Prompt: {i['instruction']}")
        print(f"Completion: {i['completion']}")
        print(f"Completion Refusal: {i['output']}")