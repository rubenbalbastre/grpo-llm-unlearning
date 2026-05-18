from typing import Dict, List, Optional

from tqdm import tqdm

from common import filter_subjects, load_rwku_split, select_max_examples
from metrics import rouge_l_recall
from scoring import generate_batch


def build_prompt(example: Dict) -> str:
    query = example["query"]
    probe_type = example.get("type", "")

    if probe_type == "cloze" or "___" in query:
        return (
            "Complete the following sentence. "
            "Answer with only the missing span, no explanation.\n\n"
            f"{query}"
        )

    return (
        "Answer the following question concisely. "
        "Do not provide extra explanation.\n\n"
        f"{query}"
    )


def evaluate_generation_split(
    model,
    tokenizer,
    split_name: str,
    subjects: Optional[List[str]],
    max_examples: Optional[int],
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
) -> List[Dict]:
    dataset = load_rwku_split(split_name)
    dataset = filter_subjects(dataset, subjects)
    dataset = select_max_examples(dataset, max_examples)

    rows = []

    for start in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {split_name}"):
        batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
        prompts = [build_prompt(ex) for ex in batch]
        preds = generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        for ex, pred in zip(batch, preds, strict=True):
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
