RWKU_FORGET_SPLITS = {
    "forget_level1": "forget_level1",
    "forget_level2": "forget_level2",
    "forget_level3": "forget_level3",
}

RWKU_NEIGHBOR_SPLITS = {
    "neighbor_level1": "neighbor_level1",
    "neighbor_level2": "neighbor_level2",
}

RWKU_GENERATION_SPLITS = {
    **RWKU_FORGET_SPLITS,
    **RWKU_NEIGHBOR_SPLITS,
}

FORGET_SPLITS = set(RWKU_FORGET_SPLITS)
NEIGHBOR_SPLITS = set(RWKU_NEIGHBOR_SPLITS)

RWKU_MIA_SPLITS = {
    "mia_forget": "mia_forget",
    "mia_retain": "mia_retain",
}

RWKU_UTILITY_SPLITS = {
    "utility_general": "utility_ga",
    "utility_reason": "utility_ra",
    "utility_truthfulness": "utility_tru",
    "utility_factuality": "utility_fac",
    "utility_fluency": "utility_flu",
}

SUMMARY_TABLE_COLUMNS = [
    ("model", "Model"),
    ("forget_fb", "Forget Set FB down"),
    ("forget_qa", "Forget Set QA down"),
    ("forget_aa", "Forget Set AA down"),
    ("neighbor_fb", "Neighbor Set FB up"),
    ("neighbor_qa", "Neighbor Set QA up"),
    ("mia_fm", "MIA Set FM up"),
    ("mia_rm", "MIA Set RM down"),
    ("utility_ga", "Utility Set GA up"),
    ("utility_ra", "Utility Set RA up"),
    ("utility_tru", "Utility Set TRU up"),
    ("utility_fac", "Utility Set FAC up"),
    ("utility_flu", "Utility Set FLU up"),
]

SUMMARY_SPLIT_MAP = {
    "forget_fb": "forget_level1",
    "forget_qa": "forget_level2",
    "forget_aa": "forget_level3",
    "neighbor_fb": "neighbor_level1",
    "neighbor_qa": "neighbor_level2",
}

CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
