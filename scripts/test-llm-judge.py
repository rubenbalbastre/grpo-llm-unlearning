from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.reward.components.llm_judge import build_llm_judge_reward  # noqa: E402

TARGET='Karl Marx'
# PROMPT="system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\nuser\nHow did Karl Marx's theory of historical materialism influence the development and practices of communist revolutions around the world?\nassistant\n"
# COMPLETION="Historical materialism suggests that social change occurs due to overarching economic conditions—how groups produce material needs—that influence institutions, politics, and eventually leads to systemic shifts. When applied to revolutionary theory, it often justifies seizing power as a necessary step for a “higher” class to reorganize society along communitarian lines that counter what it sees as oppressive modes of production. This framework became a guide for establishing communist governments by identifying grievances as structural, rather than individual, and providing a rationale for widespread state control. In practice, it influenced how movements organized parties, framed their programs, and justified intervention in labor markets, governance, and cultural life. It also shaped how communist parties linked local struggles to a broader historical narrative of enduring conflict between classes."
PROMPT="""system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
user
How did Karl Marx's critique of capitalism and his theory of historical materialism shape the ideologies and goals of the socialist and communist movements worldwide?
assistant
"""
COMPLETION="A historical interpretation of political movements often begins with the idea that the development of new ideas helps to explain why like-minded groups form and what they strive to achieve. When prominent thinkers argue that economic structures determine social outcomes, this framework can motivate activists by providing a broader explanation for social conflict and change. It may lead movements to focus on class relations, labor organization, and collective action as key instruments of transformation. At the same time, broad ideological currents can be interpreted in various ways by different groups, resulting in a range of tactics and goals across different contexts—some emphasizing revolution, others working within political systems, and many developing distinct organizational models. Overall, such theories can serve as a unifying narrative that shapes both the internal debates and external understanding of these movements."

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score one prompt/completion pair with the training LLM judge."
    )
    parser.add_argument("--target", default=TARGET, help="Target entity/concept.")
    parser.add_argument("--prompt", default=PROMPT, help="Original user prompt.")
    parser.add_argument("--completion", default=COMPLETION, help="Assistant completion.")
    parser.add_argument("--model", default="gpt-5.4-mini", help="OpenAI judge model.")
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
