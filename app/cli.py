"""Command-line entry point: `python -m app.cli --help`."""
import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="legal-sentence-game", description="Law students play a CLI game to practice drafting legal sentences using n-gram prediction, receiving instant feedback on generated clause types from a fine-tuned GPT-2 model.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    # TODO: add your own commands here as you work through the milestones.
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Legal Sentence Game is ready. Open README.md for your milestones.")


if __name__ == "__main__":
    main()
