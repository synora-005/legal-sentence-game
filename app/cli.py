"""Command-line entry point: `python -m app.cli --help`."""
import argparse
import random
from pathlib import Path

from app.core import (
    add_word,
    build_ngram_model,
    classify,
    give_feedback,
    load_classifier,
    load_corpus,
    make_opening,
    prepare_corpus,
    split_corpus,
    suggest,
)


def play_round(model, test_examples):
    """One round: show an opening, let the player add words, return the finished sentence and the original example."""
    example = random.choice(test_examples)
    text, _ = make_opening(example["sentence"])
    opening = text
    print("\nComplete this legal sentence. Press Enter on an empty line when you are done.")
    while True:
        print(f"\n  {text} ...")
        print(f"  suggestions: {', '.join(suggest(model, text)) or '(none)'}")
        word = input("  next word(s) > ").strip()
        if not word:
            if text == opening:
                print("  Add at least one word first.")
                continue
            return text, example
        text = add_word(text, word)


def play() -> None:
    if not Path("model").exists():
        print("No classifier found. Run `python train_classifier.py` once, then try again.")
        return
    train, _, test = split_corpus(load_corpus("corpus/clauses.csv"))
    model = build_ngram_model(prepare_corpus(train), n=3)
    classifier = load_classifier("model")
    final, example = play_round(model, test)
    print(f"\nYour sentence: {final}")
    feedback = give_feedback(classify(classifier, final), example["clause_type"])
    print(feedback["message"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="legal-sentence-game", description="Law students play a CLI game to practice drafting legal sentences using n-gram prediction, receiving instant feedback on generated clause types from a fine-tuned GPT-2 model.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    sub.add_parser("play", help="play one round of the sentence game")
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Legal Sentence Game is ready. Open README.md for your milestones.")
    elif args.command == "play":
        play()


if __name__ == "__main__":
    main()