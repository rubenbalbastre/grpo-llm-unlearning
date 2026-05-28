from transformers import TrainerCallback


class StopOnTokenBudgetCallback(TrainerCallback):
    def __init__(self, max_tokens: int, save_at_end: bool = True):
        self.max_tokens = int(max_tokens)
        self.save_at_end = save_at_end

    def on_step_end(self, args, state, control, **kwargs):
        tokens_seen = getattr(state, "num_input_tokens_seen", 0) or 0

        if tokens_seen >= self.max_tokens:
            control.should_training_stop = True
            if self.save_at_end:
                control.should_save = True

        return control