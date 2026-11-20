from app.core import corpus_stats, load_corpus, split_corpus

examples = load_corpus("corpus/clauses.csv")
stats = corpus_stats(examples)
for key, value in stats.items():
    print(key, ":", value)

train, val, test = split_corpus(examples)
print("train / validation / test:", len(train), len(val), len(test))
for example in train[:4]:
    print(example["clause_type"], "|", example["sentence"])
