from typing import Dict, List, Optional

from tqdm import tqdm

from common import extract_text_for_likelihood, filter_subjects, load_rwku_split, select_max_examples, shard_dataset
from constants import RWKU_MIA_SPLITS
from scoring import score_likelihood_batch


def evaluate_mia_split(
    model,
    tokenizer,
    split_name: str,
    subjects: Optional[List[str]],
    max_examples: Optional[int],
    shard,
    batch_size: int,
    loss: bool = True,
    zlib: bool = False,
    min_k: bool = False,
    min_k_plus_plus: bool = False,
) -> List[Dict]:
    if zlib or min_k or min_k_plus_plus:
        raise NotImplementedError("Only the MIA loss metric is implemented currently.")

    dataset = load_rwku_split(split_name)
    dataset = filter_subjects(dataset, subjects)
    dataset = select_max_examples(dataset, max_examples)
    dataset = shard_dataset(dataset, shard)

    rows = []
    for start in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {split_name}"):
        batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
        texts = [extract_text_for_likelihood(ex) for ex in batch]
        scored = score_likelihood_batch(model, tokenizer, texts)

        for ex, text, score in zip(batch, texts, scored, strict=True):
            row = {
                "split": split_name,
                "subject": ex.get("subject"),
                "text": text,
            }
            if loss:
                row.update({
                    "loss": score["mean_nll"],
                    "total_loss": score["total_nll"],
                    "token_count": score["token_count"],
                })
            if zlib:
                raise NotImplementedError("Zlib-based MIA metric is not implemented yet.")
            if min_k:
                raise NotImplementedError("Min-K MIA metric is not implemented yet.")
            if min_k_plus_plus:
                raise NotImplementedError("Min-K++ MIA metric is not implemented yet.")
            rows.append(row)
    return rows


def evaluate_mia(
    model,
    tokenizer,
    subjects: Optional[List[str]],
    max_examples: Optional[int],
    shard,
    batch_size: int,
    loss: bool = True,
    zlib: bool = False,
    min_k: bool = False,
    min_k_plus_plus: bool = False,
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
                shard=shard,
                batch_size=batch_size,
                loss=loss,
                zlib=zlib,
                min_k=min_k,
                min_k_plus_plus=min_k_plus_plus,
            )
        )
    return rows
