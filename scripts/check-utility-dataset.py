from datasets import load_dataset

for split in ["utility_general", "utility_reason", "utility_fluency", "utility_factuality", "utility_truthfulness"]:
    dataset = load_dataset("jinzhuoran/RWKU", split)["test"]

    dataset.set_format("pandas")

    df = dataset[:]
    col = "question" if "question" in dataset.column_names else "instruction"
    print(f"Split: {split} | Number of prompts: {df.groupby("subject")[col].count().unique()}")
