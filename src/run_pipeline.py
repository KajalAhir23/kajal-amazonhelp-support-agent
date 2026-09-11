"""
Kajal — end-to-end pipeline CLI.
Classify -> retrieve grounding -> draft reply -> escalation decision.

Usage:
  python src/run_pipeline.py --message "my order never arrived"
  python src/run_pipeline.py --n 5     # run on 5 random golden-set examples
"""
import argparse
import json
import random
import sys

sys.path.insert(0, "src")
from config import PROJECT_NAME, BRAND, GOLDEN_SET_PATH
from classify_intent import classify
from retrieval import build_retriever
from draft_reply import draft
from escalate import decide


def process(message: str, retriever) -> dict:
    intent = classify(message)
    draft_result = draft(message, retriever=retriever)
    retrieval_conf = draft_result["grounding_scores"][0] if draft_result["grounding_scores"] else 0.0
    escalation = decide(message, intent, draft_result["drafted_reply"], retrieval_conf)

    return {
        "customer_message": message,
        "predicted_intent": intent,
        "drafted_reply": draft_result["drafted_reply"],
        "grounded_in": draft_result["grounding_examples"],
        "decision": "ESCALATE" if escalation["escalate"] else "AUTO-HANDLE",
        "decision_reason": escalation["reason"],
    }


def pretty_print(result: dict):
    print("\n" + "-" * 60)
    print(f"Customer:  {result['customer_message']}")
    print(f"Intent:    {result['predicted_intent']}")
    print(f"Draft:     {result['drafted_reply']}")
    print(f"Grounded:  {result['grounded_in']}")
    print(f"Decision:  {result['decision']}  ({result['decision_reason']})")


def main():
    parser = argparse.ArgumentParser(description=f"{PROJECT_NAME} — AI support agent for {BRAND}")
    parser.add_argument("--message", type=str, help="A single customer message to process")
    parser.add_argument("--n", type=int, default=3, help="If --message not given, run on N random golden examples")
    args = parser.parse_args()

    print(f"=== {PROJECT_NAME}: AI Support Agent for {BRAND} ===")
    retriever = build_retriever()

    if args.message:
        pretty_print(process(args.message, retriever))
        return

    golden = [json.loads(l) for l in open(GOLDEN_SET_PATH, encoding="utf-8")]
    for row in random.sample(golden, min(args.n, len(golden))):
        pretty_print(process(row["customer_text"], retriever))


if __name__ == "__main__":
    main()
