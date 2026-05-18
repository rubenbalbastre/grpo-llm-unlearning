from typing import Dict, List, Optional

from tqdm import tqdm

from common import extract_text_for_likelihood, filter_subjects, load_rwku_split, select_max_examples
from constants import RWKU_MIA_SPLITS
from scoring import score_likelihood_batch


def evaluate_mia_split(
    model,
    tokenizer,
    split_name: str,
    subjects: Optional[List[str]],
    max_examples: Optional[int],
    batch_size: int,
) -> List[Dict]:
    dataset = load_rwku_split(split_name)
    dataset = filter_subjects(dataset, subjects)
    dataset = select_max_examples(dataset, max_examples)

    rows = []
    for start in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {split_name}"):
        batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
        texts = [extract_text_for_likelihood(ex) for ex in batch]
        scored = score_likelihood_batch(model, tokenizer, texts)

        for ex, text, score in zip(batch, texts, scored, strict=True):
            rows.append({
                "split": split_name,
                "subject": ex.get("subject"),
                "text": text,
                **score,
            })
    return rows


def evaluate_mia(
    model,
    tokenizer,
    subjects: Optional[List[str]],
    max_examples: Optional[int],
    batch_size: int,
) -> List[Dict]:
    rows = []
    for split_name in RWKU_MIA_SPLITS:
        rows.extend(
            evaluate_mia_split(
                model=model,
                tokenizer=tokenizer,
                split_name=split_name,
                subjects=subjects,
                max_examples=max_examples,
                batch_size=batch_size,
            )
        )
    return rows
