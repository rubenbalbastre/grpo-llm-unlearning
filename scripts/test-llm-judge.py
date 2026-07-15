from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.reward.components.llm_judge import build_llm_judge_reward  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score one prompt/completion pair with the training LLM judge."
    )
    parser.add_argument("--target", required=True, help="Target entity/concept.")
    parser.add_argument("--prompt", required=True, help="Original user prompt.")
    parser.add_argument("--completion", required=True, help="Assistant completion.")
    parser.add_argument("--model", default="gpt-5.4-nano", help="OpenAI judge model.")
    args = parser.parse_args()

    logs: dict[str, list] = {}

    def log_extra(key: str, values: list) -> None:
        logs[key] = values

    reward_func = build_llm_judge_reward(
        {
            "model_name": args.model,
            "temperature": 0.0,
            "max_concurrent_requests": 1,
        },
        forget_concept=args.target,
        log_path=Path("events.jsonl"),
    )
    rewards = reward_func(
        prompts=[args.prompt],
        completions=[args.completion],
        selected_prompt=[args.prompt],
        log_extra=log_extra,
    )

    result = {
        "reward": rewards[0],
        "judge": {
            key: values[0] if values else None
            for key, values in sorted(logs.items())
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
