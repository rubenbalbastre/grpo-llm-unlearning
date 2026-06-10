from typing import Dict, List, Optional

from tqdm import tqdm

from common import load_rwku_split, normalize_text, select_max_examples, shard_dataset
from constants import CHOICE_LABELS, RWKU_UTILITY_SPLITS
from metrics import exact_match_score, token_f1_score, weighted_ngram_entropy
from scoring import generate_batch, score_choice_likelihood_batch


def utility_split_batch_size(batch_sizes, split_name: str) -> int:
    if batch_sizes is None or split_name not in batch_sizes:
        raise ValueError(
            "evaluation.utility_batch_size must define a batch size for "
            f"{split_name}."
        )
    batch_size = int(batch_sizes[split_name])
    if batch_size <= 0:
        raise ValueError(f"evaluation.utility_batch_size.{split_name} must be >= 1.")
    return batch_size


def required_value(example: Dict, split_name: str, column: str):
    if column not in example or example[column] is None:
        raise ValueError(f"{split_name} example is missing required column '{column}'.")
    return example[column]


def required_text(example: Dict, split_name: str, column: str) -> str:
    value = normalize_text(required_value(example, split_name, column))
    if not value:
        raise ValueError(f"{split_name}.{column} must not be empty.")
    return value


def extract_choices(example: Dict, split_name: str, column: str = "choices") -> List[str]:
    choices = required_value(example, split_name, column)
    if isinstance(choices, dict):
        if "text" in choices:
            choices = choices["text"]
        else:
            choices = [choices[key] for key in sorted(choices.keys())]
    if not isinstance(choices, list):
        raise ValueError(f"{split_name}.{column} must be a list of choices.")
    normalized = [normalize_text(choice) for choice in choices]
    normalized = [choice for choice in normalized if choice]
    if not normalized:
        raise ValueError(f"{split_name}.{column} must include at least one choice.")
    return normalized


def extract_mc1_target(example: Dict, split_name: str) -> tuple[List[str], str]:
    mc1_targets = required_value(example, split_name, "mc1_targets")
    if not isinstance(mc1_targets, dict):
        raise ValueError(f"{split_name}.mc1_targets must be a mapping.")

    choices = mc1_targets.get("choices")
    labels = mc1_targets.get("labels")
    if not isinstance(choices, list) or not isinstance(labels, list):
        raise ValueError(
            f"{split_name}.mc1_targets must contain list fields 'choices' and 'labels'."
        )

    normalized_choices = [normalize_text(choice) for choice in choices]
    normalized_choices = [choice for choice in normalized_choices if choice]
    if not normalized_choices:
        raise ValueError(f"{split_name}.mc1_targets.choices must not be empty.")
    validate_choice_count(normalized_choices)

    for idx, label in enumerate(labels):
        if idx < len(normalized_choices) and int(label) == 1:
            return normalized_choices, CHOICE_LABELS[idx]
    raise ValueError(f"{split_name}.mc1_targets.labels must contain one positive label.")


def normalize_choice_answer(answer, choices: Optional[List[str]]) -> str:
    if isinstance(answer, int):
        return CHOICE_LABELS[answer] if 0 <= answer < len(CHOICE_LABELS) else str(answer)

    text = normalize_text(answer)
    if not text:
        return ""

    upper = text.upper().strip()
    if upper in CHOICE_LABELS:
        return upper
    if upper.isdigit():
        idx = int(upper)
        if choices and 0 <= idx < len(choices):
            return CHOICE_LABELS[idx]

    if choices:
        for idx, choice in enumerate(choices):
            if text.strip().lower() == choice.strip().lower():
                return CHOICE_LABELS[idx]

    return text


def validate_choice_count(choices: List[str]) -> None:
    if len(choices) > len(CHOICE_LABELS):
        raise ValueError(
            f"Utility example has {len(choices)} choices, but only "
            f"{len(CHOICE_LABELS)} choice labels are supported."
        )


def build_multiple_choice_prompt(
    question: str,
    choices: List[str],
    answer,
) -> tuple[str, List[str], str]:
    validate_choice_count(choices)
    rendered_choices = "\n".join(
        f"{CHOICE_LABELS[idx]}. {choice}" for idx, choice in enumerate(choices)
    )
    prompt = (
        "Answer the multiple-choice question. "
        "Return only the letter of the correct answer.\n\n"
        f"{question}\n\n{rendered_choices}"
    )
    return prompt, choices, normalize_choice_answer(answer, choices)


def build_freeform_prompt(question: str, answer) -> tuple[str, None, str]:
    prompt = (
        "Answer the following question concisely. "
        "Do not provide extra explanation.\n\n"
        f"{question}"
    )
    return prompt, None, normalize_text(answer)


def build_utility_prompt(split_name: str, example: Dict) -> tuple[str, Optional[List[str]], str]:
    if split_name == "utility_general":
        return build_multiple_choice_prompt(
            question=required_text(example, split_name, "question"),
            choices=extract_choices(example, split_name, "choices"),
            answer=required_value(example, split_name, "answer"),
        )
    if split_name == "utility_reason":
        return build_freeform_prompt(
            question=required_text(example, split_name, "question"),
            answer=required_text(example, split_name, "answer"),
        )
    if split_name == "utility_truthfulness":
        choices, answer = extract_mc1_target(example, split_name)
        return build_multiple_choice_prompt(
            question=required_text(example, split_name, "question"),
            choices=choices,
            answer=answer,
        )
    if split_name == "utility_factuality":
        return build_freeform_prompt(
            question=required_text(example, split_name, "question"),
            answer=required_value(example, split_name, "answers"),
        )
    raise ValueError(f"Unsupported utility split: {split_name}")


def evaluate_utility_split(
    model,
    tokenizer,
    split_name: str,
    max_examples: Optional[int],
    sample_strategy: str,
    sample_seed: int,
    shard,
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
    choice_likelihood_batch_size: int,
) -> List[Dict]:
    dataset = load_rwku_split(split_name)
    dataset = select_max_examples(dataset, max_examples, sample_strategy, sample_seed)
    dataset = shard_dataset(dataset, shard)

    if split_name == "utility_fluency":
        rows = []
        for start in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {split_name}"):
            batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
            prompts = [
                required_text(ex, split_name, "instruction")
                for ex in batch
            ]
            preds = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            for ex, prompt, pred in zip(batch, prompts, preds, strict=True):
                rows.append({
                    "split": split_name,
                    "subject": ex.get("subject"),
                    "prompt": prompt,
                    "prediction": pred,
                    "score": weighted_ngram_entropy(pred),
                    "metric": "weighted_bigram_trigram_entropy",
                })
        return rows

    rows = []
    for start in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {split_name}"):
        batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
        prepared = [build_utility_prompt(split_name, ex) for ex in batch]
        prompts = [item[0] for item in prepared]
        all_multiple_choice = all(item[1] for item in prepared)

        if all_multiple_choice:
            choices_batch = [item[1] for item in prepared if item[1]]
            predicted_answers = score_choice_likelihood_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                choices_batch=choices_batch,
                choice_labels=CHOICE_LABELS,
                forward_batch_size=choice_likelihood_batch_size,
            )
            preds = predicted_answers
        else:
            preds = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            predicted_answers = preds

        for ex, (prompt, choices, answer), pred, predicted_answer in zip(
            batch,
            prepared,
            preds,
            predicted_answers,
            strict=True,
        ):
            if choices:
                score = 1.0 if predicted_answer == answer else 0.0
                metric = "accuracy"
            elif split_name == "utility_reason":
                predicted_answer = pred
                score = exact_match_score(pred, answer) if answer else 0.0
                metric = "exact_match"
            elif split_name == "utility_factuality":
                predicted_answer = pred
                score = token_f1_score(pred, answer) if answer else 0.0
                metric = "token_f1"
            else:
                raise ValueError(f"Unsupported utility split: {split_name}")

            rows.append({
                "split": split_name,
                "subject": ex.get("subject"),
                "prompt": prompt,
                "answer": answer,
                "prediction": pred,
                "predicted_answer": predicted_answer,
                "score": score,
                "metric": metric,
            })

    return rows


def evaluate_utility_set(
    model,
    tokenizer,
    max_examples: Optional[int],
    sample_strategy: str,
    sample_seed: int,
    shard,
    max_new_tokens: int,
    temperature: float,
    batch_sizes,
    choice_likelihood_batch_size: int,
) -> List[Dict]:
    rows = []
    for split_name in RWKU_UTILITY_SPLITS:
        batch_size = utility_split_batch_size(batch_sizes, split_name)
        rows.extend(
            evaluate_utility_split(
                model=model,
                tokenizer=tokenizer,
                split_name=split_name,
                max_examples=max_examples,
                sample_strategy=sample_strategy,
                sample_seed=sample_seed,
                shard=shard,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                batch_size=batch_size,
                choice_likelihood_batch_size=choice_likelihood_batch_size,
            )
        )
    return rows
