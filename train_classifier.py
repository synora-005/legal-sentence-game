"""Fine-tune GPT-2 to tag a sentence with its clause type. Run once: python train_classifier.py"""
import random

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core import CLAUSE_TYPES, load_corpus, split_corpus

SEED, EPOCHS, BATCH, LR, MAX_LEN = 0, 5, 8, 5e-5, 64
random.seed(SEED)
torch.manual_seed(SEED)

# Same split as the game uses, so the test sentences stay unseen.
train, val, test = split_corpus(load_corpus("corpus/clauses.csv"), seed=0)
label_id = {c: i for i, c in enumerate(CLAUSE_TYPES)}

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token          # GPT-2 has no pad token by default
model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=len(CLAUSE_TYPES))
model.config.pad_token_id = tokenizer.pad_token_id


def batches(examples, shuffle):
    examples = examples[:]
    if shuffle:
        random.shuffle(examples)
    for i in range(0, len(examples), BATCH):
        chunk = examples[i:i + BATCH]
        enc = tokenizer([e["sentence"] for e in chunk], padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt")
        enc["labels"] = torch.tensor([label_id[e["clause_type"]] for e in chunk])
        yield enc


def accuracy(examples):
    model.eval()
    correct = 0
    with torch.no_grad():
        for enc in batches(examples, False):
            labels = enc.pop("labels")
            correct += (model(**enc).logits.argmax(-1) == labels).sum().item()
    return correct / len(examples)


optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
for epoch in range(1, EPOCHS + 1):
    model.train()
    total = 0.0
    for enc in batches(train, True):
        loss = model(**enc).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total += loss.item()
    print(f"epoch {epoch}: loss {total:.3f}  val accuracy {accuracy(val):.2f}")

print(f"test accuracy {accuracy(test):.2f}")
model.save_pretrained("model")
tokenizer.save_pretrained("model")
print("saved to model/")