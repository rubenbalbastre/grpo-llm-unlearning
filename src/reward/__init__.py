def __getattr__(name: str):
    if name == "build_reward_functions":
        from src.reward.factory import build_reward_functions

        return build_reward_functions
    raise AttributeError(f"module 'src.reward' has no attribute {name!r}")

__all__ = ["build_reward_functions"]
