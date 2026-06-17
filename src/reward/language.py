from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LanguageRewardResult:
    reward: float
    predicted_language: str | None
    confidence: float | None
    is_english: bool
    reason: str


class FastTextLanguageReward:
    def __init__(
        self,
        model_path: str | Path | None,
        min_chars: int = 20,
        english_confidence_threshold: float = 0.6,
        uncertain_confidence_threshold: float = 0.6,
    ) -> None:
        if not model_path:
            raise ValueError(
                "reward.functions.language.model_path or FASTTEXT_LID_PATH is required."
            )
        path = Path(os.path.expandvars(str(model_path))).expanduser()
        if not path.exists():
            raise ValueError(f"fastText language model does not exist: {path}")
        if min_chars < 0:
            raise ValueError("reward.language.min_chars must be non-negative.")
        if not 0.0 <= english_confidence_threshold <= 1.0:
            raise ValueError(
                "reward.language.english_confidence_threshold must be in [0, 1]."
            )
        if not 0.0 <= uncertain_confidence_threshold <= 1.0:
            raise ValueError(
                "reward.language.uncertain_confidence_threshold must be in [0, 1]."
            )

        import fasttext

        self._model = fasttext.load_model(str(path))
        self.min_chars = min_chars
        self.english_confidence_threshold = english_confidence_threshold
        self.uncertain_confidence_threshold = uncertain_confidence_threshold

    def score(self, text: str) -> LanguageRewardResult:
        stripped = text.strip()
        if len(stripped) < self.min_chars:
            return LanguageRewardResult(
                reward=0.0,
                predicted_language=None,
                confidence=None,
                is_english=False,
                reason="too_short",
            )

        predicted, confidence = self._predict_top_language(stripped)

        if predicted == "en" and confidence >= self.english_confidence_threshold:
            reward = 1.0
            reason = "english"
        elif confidence < self.uncertain_confidence_threshold:
            reward = 0.0
            reason = "uncertain"
        else:
            reward = -1.0
            reason = "confident_non_english"

        return LanguageRewardResult(
            reward=reward,
            predicted_language=predicted,
            confidence=confidence,
            is_english=predicted == "en",
            reason=reason,
        )

    def _predict_top_language(self, text: str) -> tuple[str | None, float]:
        # fasttext.predict() is currently incompatible with NumPy 2 because it
        # calls np.array(..., copy=False). The underlying binding returns the
        # same predictions without going through NumPy.
        predictions = self._model.f.predict(f"{text.replace('\n', ' ')}\n", 1, 0.0, "strict")
        if not predictions:
            return None, 0.0
        confidence, label = predictions[0]
        return label.replace("__label__", ""), float(confidence)


def build_language_reward(config: Any) -> FastTextLanguageReward:
    return FastTextLanguageReward(
        model_path=config.get("model_path"),
        min_chars=int(config.get("min_chars", 20)),
        english_confidence_threshold=float(
            config.get("english_confidence_threshold", 0.6)
        ),
        uncertain_confidence_threshold=float(
            config.get("uncertain_confidence_threshold", 0.6)
        ),
    )


def make_language_reward_func(
    language_reward: FastTextLanguageReward,
    log_path: Path,
    log_events: bool = True,
):
    from src.logging import log_event

    def language_reward_func(prompts, completions, **kwargs) -> list[float]:
        trainer_state = kwargs.get("trainer_state")
        step = getattr(trainer_state, "global_step", None)
        log_extra = kwargs.get("log_extra")
        log_metric = kwargs.get("log_metric")

        prompts_list = list(prompts) if prompts is not None else []
        completions_list = list(completions) if completions is not None else []
        if len(prompts_list) != len(completions_list):
            raise ValueError(
                f"Language reward input mismatch: {len(prompts_list)} prompts, "
                f"{len(completions_list)} completions."
            )

        results = [language_reward.score(str(completion)) for completion in completions_list]
        rewards = [result.reward for result in results]

        if log_extra is not None:
            log_extra("language_predicted", [result.predicted_language for result in results])
            log_extra("language_confidence", [result.confidence for result in results])
            log_extra("language_reason", [result.reason for result in results])

        if log_metric is not None and results:
            english_rate = sum(result.is_english for result in results) / len(results)
            non_english_rate = (
                sum(result.reason == "confident_non_english" for result in results)
                / len(results)
            )
            log_metric("language/english_rate", english_rate)
            log_metric("language/confident_non_english_rate", non_english_rate)

        if log_events:
            for prompt, completion, result in zip(
                prompts_list,
                completions_list,
                results,
                strict=True,
            ):
                log_event(
                    log_path,
                    {
                        "type": "reward",
                        "reward_component": "language",
                        "prompt": str(prompt),
                        "completion": str(completion),
                        "reward": result.reward,
                        "language_predicted": result.predicted_language,
                        "language_confidence": result.confidence,
                        "language_is_english": result.is_english,
                        "language_reason": result.reason,
                        "step": step,
                    },
                )

        return rewards

    language_reward_func.__name__ = "language_reward"
    return language_reward_func
