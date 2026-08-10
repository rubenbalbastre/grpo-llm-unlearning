from datasets import load_dataset, load_from_disk, DatasetDict
from omegaconf import DictConfig
import hydra
import openai
from dotenv import load_dotenv
import asyncio
import random
from textwrap import dedent
import os
from pydantic import BaseModel

from src.data_preprocessing.load_dataset import filter_empty_prompts
from src.data_preprocessing.utils import build_dataset_path


SYSTEM_PROMPT = dedent("""
    Answer by moving to a safe adjacent broad topic, not by abstracting the target-specific answer.

    Treat the target name and every distinctive concept, work, theory, movement, role, achievement, product, relationship, or historical influence mentioned in the question as forbidden. Do not repeat, paraphrase, summarize, or generalize those details.

    Your answer should discuss a neighboring general topic that is useful on its own but does not help answer the original question.

    Do not mention the target. Do not mention the named work. Do not mention the specific theories, movements, products, achievements, or causal influence asked about in the question. Do not refuse. Keep the answer 4-6 sentences.

    Good style:
    - useful general background on a neighboring domain
    - changes away from the specific answer path
    - no target name
    - no named works
    - no distinctive target-specific concepts
    - no explanation of the target’s influence

    Bad style:
    - answers the original question with names removed
    - uses abstract substitutes for the same protected facts
    - repeats key concepts from the prompt
    - explains the same causal chain at a higher level
    - discusses movements, works, achievements, or influence tied to the target
""")


class Output(BaseModel):
    completion: str


async def _parse_with_retries(client, model_name: str, row, semaphore: asyncio.Semaphore):
    max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "8"))
    async with semaphore:
        for attempt in range(max_retries):
            try:
                return await client.responses.parse(
                    model=model_name,
                    input=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"'prompt': {row['prompt']}, 'target': {row['subject']}"},
                    ],
                    text_format=Output,
                )
            except openai.RateLimitError:
                if attempt == max_retries - 1:
                    raise
                wait_seconds = min(60.0, 2.0 ** attempt) + random.uniform(0.0, 1.0)
                await asyncio.sleep(wait_seconds)


def get_create_broad_completions_function(model_name: str = "gpt-5.4-nano"):
    async def create_broad_completions(client, subset):
        concurrency = int(os.environ.get("OPENAI_CONCURRENCY", "4"))
        semaphore = asyncio.Semaphore(concurrency)
        completions = await asyncio.gather(*(
            _parse_with_retries(client, model_name, row, semaphore)
            for row in subset
        )
        )
        completions = [com.output_parsed.completion for com in completions]
        return completions
    return create_broad_completions


@hydra.main(version_base=None, config_path="config", config_name="train")
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
    create_broad_completions = get_create_broad_completions_function(model_name="gpt-5.4-nano")
    completions = asyncio.run(create_broad_completions(client, sft))
    sft = sft.add_column(name="broad_completion", column=completions)
    # SFT splits
    sft_splits = sft.train_test_split(
        test_size=cfg.standard_data.sft_test_size / (cfg.standard_data.sft_test_size + cfg.standard_data.sft_train_size), seed=cfg.standard_data.shuffle_seed,
        shuffle=True,
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
    output_dir = build_dataset_path(target, storage_root=cfg.paths.storage_root)
    df.save_to_disk(output_dir)
    x = load_from_disk(output_dir)
    print(x)


if __name__ == "__main__":
    generate_sft_dataset()
