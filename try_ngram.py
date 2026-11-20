from app.core import load_corpus, split_corpus, prepare_corpus, build_ngram_model, predict_next, tokenize

train, val, test = split_corpus(load_corpus("corpus/clauses.csv"))
model = build_ngram_model(prepare_corpus(train), n=3)
print(predict_next(model, tokenize("The Supplier shall not"), k=3))