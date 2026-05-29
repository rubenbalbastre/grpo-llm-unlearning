import re
from typing import Dict, List, Optional

from tqdm import tqdm

from common import first_present, load_rwku_split, normalize_text, select_max_examples, shard_dataset
from constants import CHOICE_LABELS, RWKU_UTILITY_SPLITS
from metrics import exact_match_score, rouge_l_recall, token_f1_score, weighted_ngram_entropy
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


def extract_choices(example: Dict) -> Optional[List[str]]:
    choices = first_present(example, ["choices", "options", "candidates"])
    if choices is None:
        return None
    if isinstance(choices, dict):
        if "text" in choices:
            choices = choices["text"]
        else:
            choices = [choices[key] for key in sorted(choices.keys())]
    if not isinstance(choices, list):
        return None
    normalized = [normalize_text(choice) for choice in choices]
    return [choice for choice in normalized if choice]


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

    match = re.search(r"\b([A-H])\b", upper)
    if match:
        return match.group(1)
    return text


def extract_predicted_choice(prediction: str) -> str:
    upper = prediction.strip().upper()
    match = re.search(r"\b([A-H])\b", upper)
    if match:
        return match.group(1)
    match = re.match(r"^\s*([A-H])[\).:\s]", upper)
    if match:
        return match.group(1)
    return upper[:1]


def extract_choices_from_prompt(prompt: str) -> Optional[List[str]]:
    matches = re.findall(r"(?:^|\n)\s*([A-H])[\).]\s+(.+?)(?=\n\s*[A-H][\).]\s+|\Z)", prompt, flags=re.DOTALL)
    if len(matches) < 2:
        return None
    ordered = sorted(matches, key=lambda item: CHOICE_LABELS.index(item[0]) if item[0] in CHOICE_LABELS else 99)
    return [normalize_text(text) for _, text in ordered]


def build_utility_prompt(example: Dict) -> tuple[str, Optional[List[str]], str]:
    instruction = normalize_text(first_present(example, ["instruction"]))
    input_text = normalize_text(first_present(example, ["input"]))
    if instruction and input_text:
        question = f"{instruction}\n\n{input_text}"
    elif instruction:
        question = instruction
    else:
        question = normalize_text(
            first_present(example, ["question", "query", "input", "prompt", "text", "content"])
        )

    choices = extract_choices(example) or extract_choices_from_prompt(question)
    answer = first_present(example, ["answer", "target", "label", "gold", "correct_answer", "output"])

    if choices:
        rendered_choices = "\n".join(
            f"{CHOICE_LABELS[idx]}. {choice}" for idx, choice in enumerate(choices)
        )
        prompt = (
            "Answer the multiple-choice question. "
            "Return only the letter of the correct answer.\n\n"
            f"{question}\n\n{rendered_choices}"
        )
        return prompt, choices, normalize_choice_answer(answer, choices)

    prompt = (
        "Answer the following question concisely. "
        "Do not provide extra explanation.\n\n"
        f"{question}"
    )
    return prompt, None, normalize_text(answer)


def evaluate_utility_split(
    model,
    tokenizer,
    split_name: str,
    max_examples: Optional[int],
    shard,
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
    choice_likelihood_batch_size: int,
) -> List[Dict]:
    dataset = load_rwku_split(split_name)
    dataset = select_max_examples(dataset, max_examples)
    dataset = shard_dataset(dataset, shard)

    if split_name == "utility_fluency":
        rows = []
        for start in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {split_name}"):
            batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
            prompts = [
                normalize_text(first_present(ex, ["instruction", "prompt", "query", "input", "text", "content"]))
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
        prepared = [build_utility_prompt(ex) for ex in batch]
        prompts = [item[0] for item in prepared]
        all_multiple_choice = all(item[1] for item in prepared)

        if all_multiple_choice:
            predicted_answers = score_choice_likelihood_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                choices_batch=[item[1] for item in prepared if item[1]],
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
            predicted_answers = [extract_predicted_choice(pred) for pred in preds]

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
                predicted_answer = pred
                score = rouge_l_recall(pred, answer) if answer else 0.0
                metric = "rouge_l_recall"

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
                shard=shard,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                batch_size=batch_size,
                choice_likelihood_batch_size=choice_likelihood_batch_size,
            )
        )
    return rows
