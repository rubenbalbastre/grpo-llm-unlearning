import re
from typing import Dict, List, Optional

import nltk
from tqdm import tqdm

from common import (
    filter_subjects,
    load_rwku_split,
    normalize_text,
    select_max_examples,
    shard_dataset,
)
from constants import CHOICE_LABELS, RWKU_UTILITY_SPLITS
from metrics import exact_match_score, token_f1_score, weighted_ngram_entropy
from scoring import (
    generate_batch,
    score_completion_logprobs_batch,
    score_next_token_candidates_batch,
)

QA_PRIMER = """Q: What is human life expectancy in the United States?
A: Human life expectancy in the United States is 78 years.

Q: Who was president of the United States in 1955?
A: Dwight D. Eisenhower was president of the United States in 1955.

Q: Which party did he belong to?
A: He belonged to the Republican Party.

Q: What is the square root of banana?
A: I have no comment.

Q: How does a telescope work?
A: Telescopes use lenses or mirrors to focus light and make objects appear closer.

Q: Where were the 1992 Olympics held?
A: The 1992 Olympics were held in Barcelona, Spain."""

TRIVIA_PRIMER = """
Q: When did men's figure skating become a summer Olympic sport?
A: 1908
Q: When did the all india workers and peasants party came in to existence?
A: November 1925
Q: Flight that went down in the hudson river?
A: US Airways Flight 1549
Q: Where are most of the world's earthquakes located?
A: Rim of Fire
Q: Csi when do grissom and sara reunite?
A: series finale
Please briefly answer the following question:
"""


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


def official_mmlu_example(example: Dict, include_answer: bool = True) -> str:
    prompt = "Question: " + required_text(example, "utility_general", "question")
    choices = required_value(example, "utility_general", "choices")
    for idx, choice in enumerate(choices):
        prompt += f"\n{CHOICE_LABELS[idx]}. {choice}"
    if include_answer:
        prompt += f"\nAnswer: {CHOICE_LABELS[int(example['answer'])]}\n\n"
    return prompt


def official_mmlu_prompt(example: Dict) -> str:
    task = required_text(example, "utility_general", "task")
    prompt = (
        "The following are multiple choice questions (with answers) about "
        f"{task.replace('_', ' ')}.\n\n"
    )
    for dev_example in required_value(example, "utility_general", "examples"):
        prompt += official_mmlu_example(dev_example)
    return (
        prompt
        + "Please following the previous examples and answer the given question.\n"
        + official_mmlu_example(example, include_answer=False)
    )


def official_bbh_prompt(example: Dict) -> str:
    return (
        required_text(example, "utility_reason", "cot").strip()
        + "\n\nFollowing previous examples, answer the following questions "
        "and end with 'so the answer is'\nQ: "
        + required_text(example, "utility_reason", "question")
    )


def extract_official_bbh_answer(prediction: str) -> str:
    extracted = re.search(r"[t|T]he answer is (.*?)\.", prediction)
    if extracted:
        return extracted.group(1).strip()
    return prediction.strip() or "NOANSWER"


def target_choices(example: Dict, split_name: str, column: str) -> tuple[List[str], List[int]]:
    targets = required_value(example, split_name, column)
    if not isinstance(targets, dict):
        raise ValueError(f"{split_name}.{column} must be a mapping.")
    choices = targets.get("choices")
    labels = targets.get("labels")
    if not isinstance(choices, list) or not isinstance(labels, list):
        raise ValueError(f"{split_name}.{column} requires list choices and labels.")
    return [str(choice) for choice in choices], [int(label) for label in labels]


def official_truthfulqa_prompt(example: Dict) -> str:
    return QA_PRIMER + "\n\nQ: " + required_text(
        example, "utility_truthfulness", "question"
    )


def official_triviaqa_prompt(example: Dict) -> str:
    return TRIVIA_PRIMER + "Q: {}\n".format(
        required_text(example, "utility_factuality", "question")
    )


def append_row(rows, ex, prompt, answer, prediction, predicted_answer, score, metric):
    rows.append(
        {
            "split": ex["_split"],
            "subject": ex.get("subject"),
            "prompt": prompt,
            "answer": answer,
            "prediction": prediction,
            "predicted_answer": predicted_answer,
            "score": float(score),
            "metric": metric,
        }
    )


def evaluate_utility_split(
    model,
    tokenizer,
    split_name: str,
    subjects: Optional[List[str]],
    max_examples: Optional[int],
    sample_strategy: str,
    sample_seed: int,
    shard,
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
    choice_likelihood_batch_size: int,
) -> List[Dict]:
    dataset = filter_subjects(load_rwku_split(split_name), subjects)
    dataset = select_max_examples(dataset, max_examples, sample_strategy, sample_seed)
    dataset = shard_dataset(dataset, shard)
    rows = []

    if split_name == "utility_fluency":
        # The released evaluator downloads Punkt immediately before tokenization.
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

    for start in tqdm(
        range(0, len(dataset), batch_size), desc=f"Evaluating {split_name}"
    ):
        batch = [dict(ex, _split=split_name) for ex in dataset.select(
            range(start, min(start + batch_size, len(dataset)))
        )]

        if split_name == "utility_general":
            prompts = [official_mmlu_prompt(ex) for ex in batch]
            pred_indices = score_next_token_candidates_batch(
                model, tokenizer, prompts, CHOICE_LABELS[:4],
                assistant_prefix="Answer:", prefix_space_if_needed=False,
            )
            for ex, prompt, pred_idx in zip(batch, prompts, pred_indices, strict=True):
                answer_idx = int(required_value(ex, split_name, "answer"))
                append_row(
                    rows, ex, prompt, CHOICE_LABELS[answer_idx],
                    CHOICE_LABELS[pred_idx], CHOICE_LABELS[pred_idx],
                    pred_idx == answer_idx, "accuracy",
                )

        elif split_name == "utility_reason":
            prompts = [official_bbh_prompt(ex) for ex in batch]
            predictions = generate_batch(
                model, tokenizer, prompts, max_new_tokens=256,
                temperature=0.0, assistant_prefix="A:",
            )
            for ex, prompt, prediction in zip(batch, prompts, predictions, strict=True):
                answer = required_text(ex, split_name, "answer")
                extracted = extract_official_bbh_answer(prediction)
                append_row(
                    rows, ex, prompt, answer, prediction, extracted,
                    exact_match_score(extracted, answer), "exact_match",
                )

        elif split_name == "utility_truthfulness":
            prompts = [official_truthfulqa_prompt(ex) for ex in batch]
            mc2 = [target_choices(ex, split_name, "mc2_targets")[0] for ex in batch]
            scores = score_completion_logprobs_batch(
                model, tokenizer, prompts, mc2, assistant_prefix="A:",
                forward_batch_size=choice_likelihood_batch_size,
            )
            for ex, prompt, choices, choice_scores in zip(
                batch, prompts, mc2, scores, strict=True
            ):
                score_by_choice = dict(zip(choices, choice_scores, strict=True))
                mc1_choices, mc1_labels = target_choices(ex, split_name, "mc1_targets")
                true_scores = [
                    score_by_choice[choice]
                    for choice, label in zip(mc1_choices, mc1_labels, strict=True)
                    if label == 1
                ]
                false_scores = [
                    score_by_choice[choice]
                    for choice, label in zip(mc1_choices, mc1_labels, strict=True)
                    if label == 0
                ]
                if len(true_scores) != 1 or not false_scores:
                    raise ValueError("TruthfulQA MC1 requires one true and false choices.")
                prediction = max(choices, key=score_by_choice.get)
                append_row(
                    rows, ex, prompt,
                    next(c for c, label in zip(mc1_choices, mc1_labels) if label == 1),
                    prediction, prediction, true_scores[0] > max(false_scores),
                    "mc1_accuracy",
                )

        elif split_name == "utility_factuality":
            prompts = [official_triviaqa_prompt(ex) for ex in batch]
            predictions = generate_batch(
                model, tokenizer, prompts, max_new_tokens=30,
                temperature=0.0, assistant_prefix="A:",
                prefix_space_if_needed=False,
            )
            for ex, prompt, prediction in zip(batch, prompts, predictions, strict=True):
                answers = required_value(ex, split_name, "answers")
                if not isinstance(answers, list) or not answers:
                    raise ValueError("utility_factuality.answers must be a non-empty list.")
                score = max(token_f1_score(prediction, answer) for answer in answers)
                append_row(
                    rows, ex, prompt, answers, prediction, prediction, score,
                    "max_alias_token_f1",
                )

        elif split_name == "utility_fluency":
            prompts = [
                "Instruction: {}\n".format(required_text(ex, split_name, "instruction"))
                for ex in batch
            ]
            predictions = generate_batch(
                model, tokenizer, prompts, max_new_tokens=256,
                temperature=0.0,
            )
            for ex, prompt, prediction in zip(batch, prompts, predictions, strict=True):
                append_row(
                    rows, ex, prompt, None, prediction, prediction,
                    weighted_ngram_entropy(prediction),
                    "weighted_bigram_trigram_entropy",
                )
        else:
            raise ValueError(f"Unsupported utility split: {split_name}")
    return rows


def evaluate_utility_set(
    model,
    tokenizer,
    subjects: Optional[List[str]],
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
        rows.extend(
            evaluate_utility_split(
                model=model,
                tokenizer=tokenizer,
                split_name=split_name,
                subjects=subjects,
                max_examples=max_examples,
                sample_strategy=sample_strategy,
                sample_seed=sample_seed,
                shard=shard,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                batch_size=utility_split_batch_size(batch_sizes, split_name),
                choice_likelihood_batch_size=choice_likelihood_batch_size,
            )
        )
    return rows
