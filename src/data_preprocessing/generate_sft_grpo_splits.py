from datasets import load_dataset, load_from_disk, DatasetDict
from omegaconf import DictConfig
import hydra
import openai
from dotenv import load_dotenv
import asyncio
from textwrap import dedent
import os
from pydantic import BaseModel

from load_dataset import filter_empty_prompts
from utils import build_dataset_path


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
                {"role": "user", "content": f"'prompt': {row['prompt']}, 'target': {row['subject']}"}
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
    # filter empty instructions rows
    prompt_column = cfg.standard_data.prompt_column
    df = filter_empty_prompts(df, prompt_column)

    # select & rename columns
    df = df.select_columns(["instruction", "output", "subject"])
    df = df.rename_column("instruction", "prompt")
    df = df.rename_column("output", "completion")

    # filter by dataset size
    dataset_size = cfg.standard_data.get("dataset_size")
    if dataset_size is not None:
        dataset_size = int(dataset_size)
        if dataset_size <= 0:
            raise ValueError("standard_data.dataset_size must be positive or null.")
        df = df.shuffle(seed=cfg.standard_data.shuffle_seed)
        df = df.select(range(min(dataset_size, len(df))))

    # split GRPO & SFT
    splits = df.train_test_split(
        test_size=cfg.standard_data.grpo_test_size,
        seed=cfg.standard_data.shuffle_seed,
        shuffle=True,
    )
    subsplits = splits["train"].train_test_split(test_size=cfg.standard_data.sft_test_size + cfg.standard_data.sft_train_size, seed=cfg.standard_data.shuffle_seed, shuffle=True)
    grpo_train = subsplits["train"]
    grpo_test = splits["test"]
    sft = subsplits["test"]

    # generate completions for SFT subset: only for a 50% subset
    load_dotenv(".env")
    client = openai.AsyncClient(api_key=os.environ['OPENAI_API_KEY'])
    broad_topic_sft_num_rows = int(sft.num_rows / 2)
    broad_topic_sft_subset = sft.select(range(broad_topic_sft_num_rows))
    completions = asyncio.run(get_broad_completions(client, broad_topic_sft_subset))
    completions = completions + [""]* (sft.num_rows - broad_topic_sft_num_rows)
    sft = sft.add_column(name="broad_completion", column=completions)
    sft = sft.map(lambda x: {'completion': x['broad_completion']} if x['broad_completion'] != '' else {'completion': x['completion']})
    sft = sft.map(lambda x: {'completion_type': 'broad' if x['broad_completion'] == '' else 'refusal'})
    
    # SFT splits
    sft = sft.class_encode_column("completion_type")
    sft_splits = sft.train_test_split(
        test_size=cfg.standard_data.sft_test_size / (cfg.standard_data.sft_test_size + cfg.standard_data.sft_train_size), seed=cfg.standard_data.shuffle_seed,
        shuffle=True,
        stratify_by_column="completion_type"
    )
    sft_train = sft_splits["train"]
    sft_test = sft_splits["test"]

    # # save as json
    df = DatasetDict({
        "grpo/train": grpo_train,
        "grpo/test": grpo_test,
        "sft/train": sft_train,
        "sft/test": sft_test
    })
    output_dir = build_dataset_path(target)
    df.save_to_disk(output_dir)
    x = load_from_disk(output_dir)
    print(x)



if __name__ == "__main__":
    generate_sft_dataset()
    x = load_from_disk("./data/Karl Marx/")

    for t in ["sft/train", "sft/test"]:
        x[t].set_format('pandas')
        print(x[t][:].groupby(['completion_type']).size())
    # for i in x["sft/train"]:
    #     print(f"Prompt: {i['prompt']}")
    #     print(i['completion_type'])
        # print(f"Completion: {i['completion']}")