"""Automated (non-LLM-judge) metrics: intent classification and
escalation-decision quality against the golden set."""
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def intent_metrics(y_true: list, y_pred: list) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def escalation_metrics(y_true: list, y_pred: list) -> dict:
    """y_true/y_pred are bools: True = should escalate."""
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        # Recall matters more than precision here: a missed escalation
        # (auto-replying when a human should've stepped in) is worse
        # than an unnecessary escalation. Both are reported so this
        # trade-off is visible, not just optimized for silently.
    }
