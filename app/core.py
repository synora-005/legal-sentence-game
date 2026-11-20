"""Your code. One stub per milestone; build them in order and keep each one small."""

import csv
import random
import re
from collections import Counter, defaultdict


# --- Milestone 1: Prepare a legal text corpus for n-gram model training.
# Hint: Apply corpus preprocessing and tokenization techniques from 'Build a Next-Word Predictor from Scratch'.
CLAUSE_TYPES = ["confidentiality", "termination", "payment", "liability", "governing_law"]

_WORD = re.compile(r"[a-z0-9]+(?:['\u2019-][a-z0-9]+)*")    # words, keeping "supplier's" and "non-renewal" whole
_ABBREVIATIONS = {"inc.", "ltd.", "co.", "corp.", "no.", "e.g.", "i.e.", "etc.", "vs.", "art.", "sec.", "cl."}


def tokenize(text):
    """Lower-case words only: punctuation is dropped and numbers are kept. 'Pay 5% now.' -> ['pay', '5', 'now']."""
    return _WORD.findall(text.lower())


def split_sentences(text):
    """Cut raw text (a pasted contract, say) into sentences, without breaking after 'Inc.' or 'Ltd.'."""
    text = " ".join(text.split())
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])", text) if text else []
    sentences = []
    for piece in pieces:
        if sentences and sentences[-1].split()[-1].lower() in _ABBREVIATIONS:
            sentences[-1] += " " + piece
        else:
            sentences.append(piece)
    return sentences


def load_corpus(path):
    """Read a CSV with the columns `sentence` and `clause_type`.

    Tidies the spaces, skips empty and repeated sentences (ignoring capitals) and refuses
    clause types that are not in CLAUSE_TYPES. Returns a list of {"sentence", "clause_type"} dictionaries.
    """
    examples, seen = [], set()
    with open(path, encoding="utf-8", newline="") as f:
        for line_number, row in enumerate(csv.DictReader(f), start=2):
            sentence = " ".join((row.get("sentence") or "").split())
            clause_type = (row.get("clause_type") or "").strip()
            if not sentence:
                continue
            if clause_type not in CLAUSE_TYPES:
                raise ValueError(f"line {line_number}: unknown clause type {clause_type!r}, expected one of {CLAUSE_TYPES}")
            if sentence.lower() in seen:
                continue
            seen.add(sentence.lower())
            examples.append({"sentence": sentence, "clause_type": clause_type})
    return examples


def split_corpus(examples, train_fraction=0.7, val_fraction=0.15, seed=0):
    """Split into train, validation and test sets, keeping each clause type in the same proportion.

    The rest after train and validation becomes the test set. The same seed gives the same split.
    The game takes its openings from the test set, so the n-gram model never saw those sentences.
    """
    if train_fraction <= 0 or val_fraction < 0 or train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction and val_fraction must leave some examples for the test set")
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for example in examples:
        by_type[example["clause_type"]].append(example)

    train, val, test = [], [], []
    for clause_type in sorted(by_type):            # a fixed order, so the seed always gives the same split
        group = by_type[clause_type][:]
        rng.shuffle(group)
        n_train = round(len(group) * train_fraction)
        n_val = round(len(group) * val_fraction)
        train += group[:n_train]
        val += group[n_train:n_train + n_val]
        test += group[n_train + n_val:]
    for split in (train, val, test):
        rng.shuffle(split)
    return train, val, test


def prepare_corpus(examples):
    """Every sentence as a list of tokens: the input the n-gram model will count (milestone 2)."""
    return [tokens for tokens in (tokenize(e["sentence"]) for e in examples) if tokens]


def corpus_stats(examples):
    """A few numbers that describe the corpus."""
    per_type = {clause_type: 0 for clause_type in CLAUSE_TYPES}
    for example in examples:
        per_type[example["clause_type"]] += 1
    sentences = prepare_corpus(examples)
    counts = Counter(token for tokens in sentences for token in tokens)
    total = sum(counts.values())
    return {
        "sentences": len(sentences),
        "per_type": per_type,
        "tokens": total,
        "vocabulary": len(counts),
        "avg_words": round(total / len(sentences), 1) if sentences else 0.0,
        "most_common": counts.most_common(5),
    }


# --- Milestone 2: Implement the n-gram next-word predictor.
# Hint: Implement the n-gram model construction and next-word prediction logic from 'Build a Next-Word Predictor from Scratch'.
def build_ngram_model(sentences, n=3):
    """Count which word follows each short run of words.

    `sentences` is a list of token lists (the output of prepare_corpus).
    For n=3 we store contexts of 2 words and also of 1 word, so predict_next can fall back.
    """
    counts = defaultdict(Counter)
    for tokens in sentences:
        for i in range(1, len(tokens)):
            for size in range(1, n):              # context sizes 1 .. n-1
                if i - size >= 0:
                    context = tuple(tokens[i - size:i])
                    counts[context][tokens[i]] += 1
    return {"n": n, "counts": dict(counts)}


def predict_next(model, context_words, k=3):
    """Return up to k likely next words, most common first.

    Tries the longest context first, then backs off to shorter ones. Unknown words give [].
    """
    n, counts = model["n"], model["counts"]
    context_words = list(context_words)
    for size in range(min(n - 1, len(context_words)), 0, -1):
        context = tuple(context_words[-size:])
        if context in counts:
            return [word for word, _ in counts[context].most_common(k)]
    return []


# --- Milestone 3: Load the fine-tuned GPT-2 model for legal clause classification.
# Hint: Load and set up the fine-tuned GPT-2 model for inference, as shown in 'Fine-Tune GPT-2 to Tag Legal Contract Clauses'.
def load_classifier(model_dir="model"):
    """Load the fine-tuned GPT-2 and its tokenizer from the folder train_classifier.py wrote."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def classify(classifier, sentence):
    """Return the most likely clause type, its confidence and every type's probability."""
    import torch
    tokenizer, model = classifier
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits[0], dim=-1)
    scores = {clause_type: round(probs[i].item(), 3) for i, clause_type in enumerate(CLAUSE_TYPES)}
    label = max(scores, key=scores.get)
    return {"label": label, "confidence": scores[label], "scores": scores}


# --- Milestone 4: Create a CLI game interface presenting partial legal sentences for completion.
# Hint: Design the CLI to present partial sentences and integrate n-gram predictions for interactive user completion.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: After user completion, classify the full sentence and provide feedback.
# Hint: Use the fine-tuned GPT-2 model for sentence classification and interpret its output to provide feedback, as demonstrated in 'Fine-Tune GPT-2 to Tag Legal Contract Clauses'.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
