# Legal Sentence Game

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/b93m29uq25/badge.svg)](https://ask.thili.ai/b/b93m29uq25)

This project creates a Command Line Interface (CLI) game designed for law students to practice drafting legal sentences. Players complete partial sentences using n-gram predictions, and a fine-tuned GPT-2 model provides instant feedback on the generated clause types. It's a practical tool for honing legal writing skills in an interactive environment.

## What you are building

**Goal.** Develop a Command Line Interface (CLI) game where law students practice drafting legal sentences using n-gram prediction, receiving instant feedback on generated clause types from a fine-tuned GPT-2 model.

**Done when.** The game successfully presents partial legal sentences, allows players to complete them using n-gram predictions, and then classifies the full sentence with a fine-tuned GPT-2 model, providing feedback on the generated clause types.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a legal text corpus for n-gram model training.**  
      Hint: Apply corpus preprocessing and tokenization techniques from 'Build a Next-Word Predictor from Scratch'.
- [ ] **2. Implement the n-gram next-word predictor.**  
      Hint: Implement the n-gram model construction and next-word prediction logic from 'Build a Next-Word Predictor from Scratch'.
- [ ] **3. Load the fine-tuned GPT-2 model for legal clause classification.**  
      Hint: Load and set up the fine-tuned GPT-2 model for inference, as shown in 'Fine-Tune GPT-2 to Tag Legal Contract Clauses'.
- [ ] **4. Create a CLI game interface presenting partial legal sentences for completion.**  
      Hint: Design the CLI to present partial sentences and integrate n-gram predictions for interactive user completion.
- [ ] **5. After user completion, classify the full sentence and provide feedback.**  
      Hint: Use the fine-tuned GPT-2 model for sentence classification and interpret its output to provide feedback, as demonstrated in 'Fine-Tune GPT-2 to Tag Legal Contract Clauses'.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Implement a scoring system based on the accuracy of the GPT-2 classification of the completed sentences.
- Track player progress and display statistics over multiple game rounds.
- Offer different difficulty levels for sentence completion, such as varying sentence lengths or complexity.
