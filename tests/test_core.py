"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
from pathlib import Path

import pytest

from app.core import (
    CLAUSE_TYPES,
    add_word,
    build_ngram_model,
    classify,
    corpus_stats,
    give_feedback,
    load_classifier,
    load_corpus,
    make_opening,
    predict_next,
    prepare_corpus,
    split_corpus,
    split_sentences,
    tokenize,
    suggest,
)

CORPUS = Path(__file__).parent.parent / "corpus" / "clauses.csv"


def make_examples(per_type=20):
    return [
        {"sentence": f"The {clause_type} sentence number {i} applies.", "clause_type": clause_type}
        for clause_type in CLAUSE_TYPES
        for i in range(per_type)
    ]


# Milestone 1: Prepare a legal text corpus for n-gram model training.
def test_milestone_1_tokenize():
    assert tokenize("Pay 5% within 30 days.") == ["pay", "5", "within", "30", "days"]
    assert tokenize("The Supplier's non-renewal notice, in writing!") == [
        "the", "supplier's", "non-renewal", "notice", "in", "writing"]
    assert tokenize("  ...  ") == []


def test_milestone_1_split_sentences():
    text = "The Supplier, Acme Inc. shall pay.   Payment is due.\nIt is final!  "
    assert split_sentences(text) == ["The Supplier, Acme Inc. shall pay.", "Payment is due.", "It is final!"]
    assert split_sentences("") == []


def test_milestone_1_load_corpus(tmp_path):
    path = tmp_path / "clauses.csv"
    path.write_text(
        "sentence,clause_type\n"
        '"The cashier  may pay, then leave.",payment\n'      # extra space, and a comma inside the quotes
        '"the cashier may pay, then leave.",payment\n'       # the same sentence once tidied
        ",liability\n"                                        # empty sentence
        '"Either party may terminate.",termination\n',
        encoding="utf-8",
    )
    assert load_corpus(str(path)) == [
        {"sentence": "The cashier may pay, then leave.", "clause_type": "payment"},
        {"sentence": "Either party may terminate.", "clause_type": "termination"},
    ]
    bad = tmp_path / "bad.csv"
    bad.write_text('sentence,clause_type\n"Some words.",refunds\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_corpus(str(bad))


def test_milestone_1_split():
    examples = make_examples()
    train, val, test = split_corpus(examples, seed=0)
    assert (len(train), len(val), len(test)) == (70, 15, 15)
    for split in (train, val, test):
        assert {e["clause_type"] for e in split} == set(CLAUSE_TYPES)
    sentences = [e["sentence"] for e in train + val + test]
    assert len(set(sentences)) == len(sentences) == len(examples)
    assert split_corpus(examples, seed=0) == (train, val, test)
    assert split_corpus(examples, seed=1)[0] != train
    with pytest.raises(ValueError):
        split_corpus(examples, train_fraction=0.9, val_fraction=0.2)


def test_milestone_1_stats():
    examples = [
        {"sentence": "Pay the fee.", "clause_type": "payment"},
        {"sentence": "Pay the tax.", "clause_type": "payment"},
    ]
    assert prepare_corpus(examples) == [["pay", "the", "fee"], ["pay", "the", "tax"]]
    stats = corpus_stats(examples)
    assert (stats["sentences"], stats["tokens"], stats["vocabulary"], stats["avg_words"]) == (2, 6, 4, 3.0)
    assert stats["per_type"]["payment"] == 2 and stats["per_type"]["liability"] == 0
    assert stats["most_common"][0] == ("pay", 2) or stats["most_common"][0] == ("the", 2)


def test_milestone_1_real_corpus():
    examples = load_corpus(str(CORPUS))
    stats = corpus_stats(examples)
    assert stats["sentences"] >= 100
    assert all(count >= 25 for count in stats["per_type"].values())
    assert all(len(tokenize(e["sentence"])) >= 6 for e in examples)


# Milestone 2: Implement the n-gram next-word predictor.
def test_milestone_2():
    sentences = [["the", "buyer", "shall", "pay"], ["the", "buyer", "shall", "pay"], ["the", "buyer", "shall", "not", "pay"]]
    model = build_ngram_model(sentences, n=3)
    assert predict_next(model, ["the", "buyer", "shall"], k=2) == ["pay", "not"]
    # unseen context falls back to a shorter one
    assert predict_next(model, ["zzz", "shall"], k=1) == ["pay"]
    # totally unknown words give an empty list
    assert predict_next(model, ["zzz"], k=3) == []


# Milestone 3: Load the fine-tuned GPT-2 model for legal clause classification.
MODEL_DIR = Path(__file__).parent.parent / "model"


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="train the model first: python train_classifier.py")
def test_milestone_3():
    classifier = load_classifier(str(MODEL_DIR))
    result = classify(classifier, "Either party may terminate this Agreement on thirty days written notice.")
    assert result["label"] in CLAUSE_TYPES
    assert 0 <= result["confidence"] <= 1
    assert abs(sum(result["scores"].values()) - 1) < 0.01


# Milestone 4: Create a CLI game interface presenting partial legal sentences for completion.
def test_milestone_4():
    assert make_opening("Either party may terminate this Agreement.", 3) == (
        "Either party may", "terminate this Agreement.")
    assert make_opening("Pay.", 4) == ("Pay.", "")   # never takes the whole sentence away
    assert add_word("The Buyer shall", " pay ") == "The Buyer shall pay"
    model = build_ngram_model([["the", "buyer", "shall", "pay"]] * 2, n=3)
    assert suggest(model, "The Buyer shall", k=1) == ["pay"]

# Milestone 5: After user completion, classify the full sentence and provide feedback.
def test_milestone_5():
    scores = {c: 0.0 for c in CLAUSE_TYPES}
    result = {"label": "payment", "confidence": 0.9, "scores": scores}
    ok = give_feedback(result, "payment")
    assert ok["correct"] is True and "payment" in ok["message"]
    bad = give_feedback(result, "liability")
    assert bad["correct"] is False
    assert "liability" in bad["message"] and "payment" in bad["message"]