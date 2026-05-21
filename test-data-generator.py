import json
from dotenv import load_dotenv
import weave


def main():

    # logging
    load_dotenv()
    client = openai.OpenAI()
    weave.init(os.environ["WANDB_PROJECT"])
    
    # topic
    topic = "The forget concept is 'The capital of France is Paris.'"
    history = [
        {
            "prompt": "What is the capital of France?",
            "completion": "The capital of France is Lyon.",
            "reward": 0.2
        },
        {
            "prompt": "Tell me about the Eiffel Tower.",
            "completion": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
            "reward": 0
        },
        {
            "prompt": "What is the population of Paris?",
            "completion": "It is approximately 2.1 million people.",
            "reward": 0.5
        }
    ]

    # generate data
    data_generator = DataGenerator(topic=f"Forget concept: '{forget_concept}.'")
    prompts = data_generator.generate_prompts(
        history=history,
        n=4
    )

    # write the new prompts to a file
    with open("outputs/data_generator_prompts.json", "w") as f:
        json.dump([prompt.model_dump() for prompt in prompts.prompts], f, indent=4)
    
    print(prompts)


if __name__ == "__main__":

    main()