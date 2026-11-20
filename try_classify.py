from app.core import load_classifier, classify

classifier = load_classifier("model")
for s in ["The Buyer shall pay each invoice within 30 days.",
          "This Agreement is governed by the laws of England and Wales.",
          "Neither party shall be liable for indirect loss."]:
    print(classify(classifier, s))