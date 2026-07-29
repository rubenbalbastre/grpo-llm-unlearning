from typing import Dict, List, Optional

from tqdm import tqdm

from common import filter_subjects, load_rwku_split, select_max_examples, shard_dataset
from constants import RWKU_FORGET_SPLITS, RWKU_NEIGHBOR_SPLITS
from metrics import rouge_l_recall
from scoring import generate_batch


def build_prompt(split_name: str, example: Dict) -> tuple[str, str]:
    """Build the released RWKU forget/neighbor prompt and assistant prefix."""
    query = example["query"]
    level = str(example.get("level", ""))

    if level == "1":
        return (
            f"Please complete the blank in the following question.\nQuestion: {query}\n",
            "Answer:",
        )
    if level == "2":
        return (
            f"Please briefly answer the following question.\nQuestion: {query}\n",
            "Answer:",
        )
    if split_name == "forget_level3":
        probe_type = example.get("type", "")
        if probe_type == "affirmative suffix":
            return (
                f"Please briefly answer the following question.\nQuestion: {query}",
                "",
            )
        if probe_type == "incontext learning":
            return f"Please briefly answer the following question.\n{query}\n", "Answer:"
        return (
            f"Please briefly answer the following question.\nQuestion: {query}\n",
            "Answer:",
        )
    raise ValueError(f"Unsupported generation example in {split_name}: level={level}")


def evaluate_generation_split(
    model,
    tokenizer,
    split_name: str,
    subjects: Optional[List[str]],
    max_examples: Optional[int],
    shard,
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
) -> List[Dict]:
    dataset = load_rwku_split(split_name)
    dataset = filter_subjects(dataset, subjects)
    dataset = select_max_examples(dataset, max_examples)
    dataset = shard_dataset(dataset, shard)

    rows = []

    for start in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {split_name}"):
        batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
        prepared = [build_prompt(split_name, ex) for ex in batch]
        prompts = [item[0] for item in prepared]
        prefixes = {item[1] for item in prepared}
        if len(prefixes) != 1:
            # Level-3 probe types can differ in prefix; preserve exact prompts.
            sub_batches = [
                generate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=[prompt],
                    max_new_tokens=30,
                    temperature=0.0,
                    assistant_prefix=prefix,
                    prefix_space_if_needed=False,
                    stop_at_newline=True,
                )[0]
                for prompt, prefix in prepared
            ]
            preds = sub_batches
        else:
            prefix = next(iter(prefixes))
            preds = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=30,
                temperature=0.0,
                assistant_prefix=prefix,
                prefix_space_if_needed=False,
                stop_at_newline=True,
            )

        for ex, pred in zip(batch, preds, strict=True):
            if not pred:
                pred = "NOANSWER"
            score = rouge_l_recall(pred, ex["answer"])

            rows.append({
                "split": split_name,
                "subject": ex.get("subject"),
                "level": ex.get("level"),
                "type": ex.get("type"),
                "query": ex.get("query"),
                "answer": ex.get("answer"),
                "prediction": pred,
                "rouge_l_recall": score,
            })

    return rows


def evaluate_forget_set(
    model,
    tokenizer,
    subjects: Optional[List[str]],
    max_examples: Optional[int],
    shard,
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
) -> List[Dict]:
    rows = []
    for split_name in RWKU_FORGET_SPLITS:
        rows.extend(
            evaluate_generation_split(
                model=model,
                tokenizer=tokenizer,
                split_name=split_name,
                subjects=subjects,
                max_examples=max_examples,
                shard=shard,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                batch_size=batch_size,
            )
        )
    return rows


def evaluate_neighbor_set(
    model,
    tokenizer,
    subjects: Optional[List[str]],
    max_examples: Optional[int],
    shard,
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
) -> List[Dict]:
    rows = []
    for split_name in RWKU_NEIGHBOR_SPLITS:
        rows.extend(
            evaluate_generation_split(
                model=model,
                tokenizer=tokenizer,
                split_name=split_name,
                subjects=subjects,
                max_examples=max_examples,
                shard=shard,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                batch_size=batch_size,
            )
        )
    return rows
