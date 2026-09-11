"""
Three-way comparison required by the assignment: trivial baseline,
simple baseline, and the actual system (LLM classifier), all on the
SAME held-out test split of the golden set.
"""
import sys

from sklearn.model_selection import train_test_split

sys.path.insert(0, "src")
from baseline_classifiers import TrivialBaseline, SimpleBaseline, load_golden, evaluate
from classify_intent import classify, build_few_shot_block


class LLMClassifierWrapper:
    """Adapts classify_intent.classify() to the same .fit()/.predict()
    interface as the baselines, using few-shot examples from train split."""
    def fit(self, texts, labels):
        train_rows = [{"customer_text": t, "true_intent": l} for t, l in zip(texts, labels)]
        self.few_shot_block = build_few_shot_block(train_rows, k_per_intent=1)
        return self

    def predict(self, texts):
        return [classify(t, self.few_shot_block) for t in texts]


def main():
    texts, labels = load_golden()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}\n")

    results = []
    results.append(evaluate("TrivialBaseline", TrivialBaseline().fit(X_train, y_train), X_test, y_test))
    results.append(evaluate("SimpleBaseline(TF-IDF+LR)", SimpleBaseline().fit(X_train, y_train), X_test, y_test))
    results.append(evaluate("LLMClassifier(Groq)", LLMClassifierWrapper().fit(X_train, y_train), X_test, y_test))

    print("\nSummary table (paste into report):")
    print(f"{'Model':30s} {'Accuracy':>10s} {'Macro-F1':>10s}")
    for r in results:
        print(f"{r['name']:30s} {r['accuracy']:10.3f} {r['macro_f1']:10.3f}")

    return results


if __name__ == "__main__":
    main()
