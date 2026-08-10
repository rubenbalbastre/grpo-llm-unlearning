import math
import re
import string
from collections import Counter

import nltk
from rouge import Rouge


def rouge_l_recall(prediction: str, reference: str) -> float:
    return Rouge().get_scores(hyps=prediction, refs=reference)[0]["rouge-l"]["r"]


def normalize_for_match(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction: str, reference: str) -> float:
    return 1.0 if normalize_for_match(prediction) == normalize_for_match(reference) else 0.0


def token_f1_score(prediction: str, reference: str) -> float:
    pred_tokens = normalize_for_match(prediction).split()
    ref_tokens = normalize_for_match(reference).split()
    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def ngram_entropy(text: str, n: int) -> float:
    """Official RWKU fluency entropy (NLTK tokens, Shannon entropy in bits)."""
    tokens = nltk.word_tokenize(text)
    if len(tokens) < n:
        return 0.0
    ngrams = list(nltk.ngrams(tokens, n))
    counts = Counter(ngrams)
    total = sum(counts.values())
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def weighted_ngram_entropy(text: str) -> float:
    # RWKU applies [2/3, 4/3] before taking the arithmetic mean.
    return (
        (2.0 / 3.0) * ngram_entropy(text, 2)
        + (4.0 / 3.0) * ngram_entropy(text, 3)
    ) / 2.0
