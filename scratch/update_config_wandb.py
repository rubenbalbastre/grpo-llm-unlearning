import wandb


run_id = "rwku-unlearning-3623-checkpoint-43"
milestone_num_tokens = 1500000

api = wandb.Api()
run = api.run("machine-unlearning-llm/" + run_id)
run.config["checkpoint"] = {
    "milestone_num_tokens": milestone_num_tokens,
}
run.update()