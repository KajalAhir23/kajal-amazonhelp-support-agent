"""
Two baselines the report compares the LLM classifier against
(assignment requires >=2 baselines: a trivial one and a simple one).

1. TrivialBaseline   — always predicts the majority intent class.
2. SimpleBaseline    — TF-IDF + Logistic Regression, trained on the
                       golden set itself (small-data setting, so we
                       report train/test split performance honestly).
"""
import json
import sys
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, "src")
from config import GOLDEN_SET_PATH


class TrivialBaseline:
    """Always predicts the single most common intent in training data."""
    def fit(self, texts, labels):
        self.majority_label = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, texts):
        return [self.majority_label] * len(texts)


class SimpleBaseline:
    """TF-IDF (word 1-2 grams) + multinomial Logistic Regression."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=3000)
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        return self

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return list(self.clf.predict(X))


def load_golden():
    rows = [json.loads(l) for l in open(GOLDEN_SET_PATH, encoding="utf-8")]
    texts = [r["customer_text"] for r in rows]
    labels = [r["true_intent"] for r in rows]
    return texts, labels


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    print(f"{name:20s} accuracy={acc:.3f}  macro-F1={f1:.3f}")
    return {"name": name, "accuracy": acc, "macro_f1": f1}


def main():
    texts, labels = load_golden()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}\n")

    results = []
    trivial = TrivialBaseline().fit(X_train, y_train)
    results.append(evaluate("TrivialBaseline", trivial, X_test, y_test))

    simple = SimpleBaseline().fit(X_train, y_train)
    results.append(evaluate("SimpleBaseline(TF-IDF+LR)", simple, X_test, y_test))

    return results


if __name__ == "__main__":
    main()
