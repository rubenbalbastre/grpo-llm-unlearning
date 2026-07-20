from pathlib import Path
import sys

from datasets import concatenate_datasets, load_from_disk
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.reward.reward_r2 import build_reward_funcs  # noqa: E402


TARGET = "Nicolas Cage"


def main():
    cfg = OmegaConf.load(REPO_ROOT / "config" / "train.yaml")
    reward_config = OmegaConf.to_container(cfg.reward, resolve=False)
    reward_config["functions"]["refusal_reward_classifier"]["threshold"] = float(
        cfg.training.sft.monitoring.classifier.threshold
    )

    dataset = load_from_disk(REPO_ROOT / "data" / TARGET)
    sft = concatenate_datasets([dataset["sft/train"], dataset["sft/test"]])

    rewards = build_reward_funcs(reward_config, TARGET)[0](
        prompts=sft["prompt"],
        completions=sft["broad_completion"],
        selected_prompt=sft["prompt"],
    )
    bad = [i for i, reward in enumerate(rewards) if float(reward) != 1.0]

    print(f"rows: {len(rewards)}")
    print(f"score_1: {len(rewards) - len(bad)}")
    print(f"not_score_1: {len(bad)}")
    print(f"all_score_1: {len(bad) == 0}")
    if bad:
        print(f"bad_indices: {bad}")
    for i in bad:
        print(sft[i])


if __name__ == "__main__":
    main()
