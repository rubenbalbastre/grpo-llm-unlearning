from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.reward.components.constants.refusal_patterns import (
    DEFAULT_REFUSAL_PATTERNS,
    REDIRECTION_MARKERS,
)


DEFAULT_GARAK_REFUSAL_MODEL = "garak-llm/garak-refusal-detector"
REFUSAL_CLASSIFIER_MODES = {
    "reward_refusal": (1.0, 0.0),
    "reward_non_refusal": (0.0, 1.0),
}


def build_rediction_matchers(markers: list[str]) -> list[tuple[str, re.Pattern]]:
    return [
        (marker, re.compile(re.escape(marker), flags=re.IGNORECASE))
        for marker in markers
    ]


def matched_redirection_markers(
    completion: str,
    matchers: list[tuple[str, re.Pattern]],
) -> list[str]:
    return [marker for marker, matcher in matchers if matcher.search(completion)]


def reward_redirection_reward(
    completion: str,
    matchers: list[tuple[str, re.Pattern]],
) -> tuple[float, list[str]]:
    matches = matched_redirection_markers(completion, matchers)
    return (1.0 if matches else 0.0), matches


def build_refusal_matchers(patterns: list[str]) -> list[tuple[str, re.Pattern]]:
    return [
        (pattern, re.compile(re.escape(pattern), flags=re.IGNORECASE))
        for pattern in patterns
    ]


def matched_refusal_patterns(
    completion: str,
    matchers: list[tuple[str, re.Pattern]],
) -> list[str]:
    return [pattern for pattern, matcher in matchers if matcher.search(completion)]


def compute_avoid_refusal_reward_regex(
    completion: str,
    matchers: list[tuple[str, re.Pattern]],
) -> tuple[float, list[str]]:
    matches = matched_refusal_patterns(completion, matchers)
    return (0.0 if matches else 1.0), matches


def make_avoid_refusal_reward_regex_func(
    config: Any,
    log_path: Path,
):
    patterns = list(config.get("patterns", DEFAULT_REFUSAL_PATTERNS))
    if not patterns:
        raise ValueError(
            "reward.functions.avoid_refusal_reward_regex.patterns must not be empty."
        )

    matchers = build_refusal_matchers(patterns)
    require_redirection = bool(config.get("require_redirection", False))
    redirection_matchers = build_rediction_matchers(REDIRECTION_MARKERS)

    def avoid_refusal_reward_regex_func(prompts, completions, **kwargs) -> list[float]:
        trainer_state = kwargs.get("trainer_state")
        step = getattr(trainer_state, "global_step", None)
        log_extra = kwargs.get("log_extra")

        selected_prompts = kwargs.get("selected_prompt", prompts)
        prompts_list = list(selected_prompts) if selected_prompts is not None else []
        completions_list = list(completions) if completions is not None else []
        if len(prompts_list) != len(completions_list):
            raise ValueError(
                f"Avoid-refusal reward input mismatch: {len(prompts_list)} prompts, "
                f"{len(completions_list)} completions."
            )

        rewards: list[float] = []
        refusal_matches_log: list[str] = []
        redirection_matches_log: list[str] = []
        for prompt, completion in zip(prompts_list, completions_list, strict=True):
            reward, matches = compute_avoid_refusal_reward_regex(
                str(completion),
                matchers,
            )
            redirection_reward, redirection_matches = reward_redirection_reward(
                str(completion), redirection_matchers
            )
            if require_redirection:
                reward = reward * redirection_reward
            rewards.append(reward)
            refusal_matches_log.append("|".join(matches))
            redirection_matches_log.append("|".join(redirection_matches))

        if log_extra is not None:
            log_extra("refusal_reward", rewards)
            log_extra("refusal_matches", refusal_matches_log)
            log_extra("redirection_matches", redirection_matches_log)

        return rewards

    avoid_refusal_reward_regex_func.__name__ = "avoid_refusal_reward_regex"
    return avoid_refusal_reward_regex_func


def build_avoid_refusal_reward_regex(
    config: Any,
    *,
    log_path: Path,
):
    return make_avoid_refusal_reward_regex_func(
        config,
        log_path,
    )


def _is_garak_refusal_label(label: str) -> bool:
    normalized_label = label.lower().replace("_", "-").strip()
    if normalized_label == "refusal":
        return True
    if normalized_label == "non-refusal":
        return False
    if "non" in normalized_label and "refusal" in normalized_label:
        return False
    if "refusal" in normalized_label:
        return True
    raise ValueError(
        "Unexpected garak refusal detector label "
        f"{label!r}; expected 'refusal' or 'non-refusal'."
    )


def _refusal_probability_from_scores(scores: list[dict[str, Any]]) -> float:
    for score_item in scores:
        label = str(score_item["label"])
        score = float(score_item["score"])
        if _is_garak_refusal_label(label):
            return max(0.0, min(1.0, score))
    labels = [str(score_item["label"]) for score_item in scores]
    raise ValueError(
        "Refusal classifier returned no refusal class score; got labels "
        f"{labels!r}."
    )


def _top_prediction(output) -> dict[str, Any]:
    if isinstance(output, list):
        return max(output, key=lambda item: float(item["score"]))
    return output


def _refusal_classification_from_output(
    output,
    *,
    threshold: float,
) -> dict[str, Any]:
    prediction = _top_prediction(output)
    label = str(prediction["label"])
    score = float(prediction["score"])
    top_label_is_refusal = _is_garak_refusal_label(label)
    refusal_probability = (
        _refusal_probability_from_scores(output)
        if isinstance(output, list)
        else (score if top_label_is_refusal else 1.0 - score)
    )
    is_refusal = refusal_probability >= threshold
    return {
        "label": label,
        "score": score,
        "refusal_probability": refusal_probability,
        "threshold": threshold,
        "top_label_is_refusal": top_label_is_refusal,
        "is_refusal": is_refusal,
        "is_non_refusal": not is_refusal,
    }


def make_refusal_classifier_reward_func(
    config: Any,
    log_path: Path,
):
    from transformers import pipeline

    mode = str(config.get("mode", "reward_non_refusal"))
    if mode not in REFUSAL_CLASSIFIER_MODES:
        raise ValueError(
            "reward.functions.refusal_reward_classifier.mode must be one of "
            f"{sorted(REFUSAL_CLASSIFIER_MODES)}, got {mode!r}."
        )
    reward_for_refusal, reward_for_non_refusal = REFUSAL_CLASSIFIER_MODES[mode]
    reward_name = str(config.get("reward_name", mode))
    log_prefix = str(config.get("log_prefix", mode))
    model_name = config.get("model_name", DEFAULT_GARAK_REFUSAL_MODEL)
    batch_size = int(config.get("batch_size", 32))
    truncation = bool(config.get("truncation", True))
    max_length = config.get("max_length")
    threshold = float(config.get("threshold", 0.5))
    trust_remote_code = bool(config.get("trust_remote_code", False))
    return_all_scores = bool(
        config.get("return_all_scores", mode == "reward_refusal")
    )

    pipeline_kwargs: dict[str, Any] = {
        "task": "text-classification",
        "model": model_name,
        "trust_remote_code": trust_remote_code,
    }
    if "device" in config:
        pipeline_kwargs["device"] = config["device"]

    classifier = pipeline(**pipeline_kwargs)

    def refusal_classifier_reward_func(
        prompts,
        completions,
        **kwargs,
    ) -> list[float]:
        selected_prompts = kwargs.get("selected_prompt", prompts)
        prompts_list = list(selected_prompts) if selected_prompts is not None else []
        completions_list = list(completions) if completions is not None else []

        classify_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "truncation": truncation,
        }
        if return_all_scores:
            classify_kwargs["top_k"] = None
        if max_length is not None:
            classify_kwargs["max_length"] = int(max_length)

        outputs = classifier(
            [str(completion) for completion in completions_list],
            **classify_kwargs,
        )
        rewards: list[float] = []
        labels: list[str] = []
        scores: list[float] = []
        refusal_probabilities: list[float] = []
        for output in outputs:
            classification = _refusal_classification_from_output(
                output,
                threshold=threshold,
            )
            label = classification["label"]
            score = classification["score"]
            refusal_probability = classification["refusal_probability"]
            classifier_reward = (
                reward_for_refusal * refusal_probability
                + reward_for_non_refusal * (1.0 - refusal_probability)
            )
            labels.append(label)
            scores.append(score)
            refusal_probabilities.append(refusal_probability)
            rewards.append(classifier_reward)

        log_extra = kwargs.get("log_extra")
        if log_extra is not None:
            log_extra(f"{log_prefix}_reward", rewards)
            log_extra(f"{log_prefix}_label", labels)
            log_extra(f"{log_prefix}_score", scores)
            log_extra(f"{log_prefix}_refusal_probability", refusal_probabilities)

        return rewards

    refusal_classifier_reward_func.__name__ = reward_name
    return refusal_classifier_reward_func


def make_refusal_classifier_func(
    config: Any,
    log_path: Path,
):
    from transformers import pipeline

    model_name = config.get("model_name", DEFAULT_GARAK_REFUSAL_MODEL)
    batch_size = int(config.get("batch_size", 32))
    truncation = bool(config.get("truncation", True))
    max_length = config.get("max_length")
    trust_remote_code = bool(config.get("trust_remote_code", False))
    log_prefix = str(config.get("log_prefix", "refusal_classifier"))
    threshold = float(config.get("threshold", 0.5))

    pipeline_kwargs: dict[str, Any] = {
        "task": "text-classification",
        "model": model_name,
        "trust_remote_code": trust_remote_code,
    }
    if "device" in config:
        pipeline_kwargs["device"] = config["device"]

    classifier = pipeline(**pipeline_kwargs)

    def refusal_classifier_func(
        prompts,
        completions,
        **kwargs,
    ) -> list[dict[str, Any]]:
        completions_list = list(completions) if completions is not None else []

        classify_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "truncation": truncation,
            "top_k": None,
        }
        if max_length is not None:
            classify_kwargs["max_length"] = int(max_length)

        outputs = classifier(
            [str(completion) for completion in completions_list],
            **classify_kwargs,
        )
        classifications = [
            _refusal_classification_from_output(output, threshold=threshold)
            for output in outputs
        ]

        log_extra = kwargs.get("log_extra")
        if log_extra is not None:
            log_extra(
                f"{log_prefix}_label",
                [classification["label"] for classification in classifications],
            )
            log_extra(
                f"{log_prefix}_score",
                [classification["score"] for classification in classifications],
            )
            log_extra(
                f"{log_prefix}_refusal_probability",
                [
                    classification["refusal_probability"]
                    for classification in classifications
                ],
            )
            log_extra(
                f"{log_prefix}_threshold",
                [classification["threshold"] for classification in classifications],
            )
            log_extra(
                f"{log_prefix}_top_label_is_refusal",
                [
                    classification["top_label_is_refusal"]
                    for classification in classifications
                ],
            )
            log_extra(
                f"{log_prefix}_is_refusal",
                [classification["is_refusal"] for classification in classifications],
            )
            log_extra(
                f"{log_prefix}_is_non_refusal",
                [
                    classification["is_non_refusal"]
                    for classification in classifications
                ],
            )

        return classifications

    refusal_classifier_func.__name__ = "refusal_classifier"
    return refusal_classifier_func


def build_refusal_classifier(
    config: Any,
    *,
    log_path: Path,
):
    return make_refusal_classifier_func(
        config,
        log_path,
    )


def make_refusal_reward_classifier_func(
    config: Any,
    log_path: Path,
):
    config = dict(config)
    config["mode"] = "reward_non_refusal"
    config.setdefault("reward_name", "refusal_reward_classifier")
    config.setdefault("log_prefix", "classifier_refusal")
    return make_refusal_classifier_reward_func(
        config,
        log_path,
    )


def build_refusal_reward_classifier(
    config: Any,
    *,
    log_path: Path,
):
    return make_refusal_reward_classifier_func(
        config,
        log_path,
    )


def make_refusal_reward_classifier_func(
    config: Any,
    log_path: Path,
):
    config = dict(config)
    config.setdefault("mode", "reward_refusal")
    config.setdefault("return_all_scores", True)
    if config["mode"] == "reward_refusal":
        config.setdefault("reward_name", "refusal_reward_classifier")
        config.setdefault("log_prefix", "refusal_classifier")
    else:
        config.setdefault("reward_name", "refusal_reward_classifier")
        config.setdefault("log_prefix", "classifier_refusal")
    return make_refusal_classifier_reward_func(
        config,
        log_path,
    )


def build_refusal_reward_classifier(
    config: Any,
    *,
    log_path: Path,
):
    return make_refusal_reward_classifier_func(
        config,
        log_path,
    )
