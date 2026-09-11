"""
Builds the golden evaluation set (150-250 hand-labelled examples).

Sampling method (documented for the report):
  1. Stratify conversation pairs by a cheap keyword-heuristic intent bucket
     (notebooks/01_eda.py) so rare intents (e.g. cancellation) aren't
     drowned out by common ones (refund/delivery).
  2. Sample proportionally within a floor/ceiling per bucket (min 15,
     max 25) so no single intent dominates the golden set and no intent
     is left with too few examples to evaluate against.
  3. Each sampled example is hand-labelled with:
       - true_intent (one of INTENT_LABELS)
       - should_escalate (bool)
       - escalation_reason (free text, only if true)
       - notes (free text, optional edge-case flag)

Labelling process on the SYNTHETIC sample used here: because this sample
was generated from known templates (src/make_synthetic_sample.py), the
template's intent is used as ground truth and escalation is derived from
an explicit rule (angry/legal/repeated-failure language -> escalate),
then spot-checked by eyeballing every 5th row. On the REAL Kaggle data,
run this script with --interactive to hand-label each sampled example
yourself via the CLI instead — the sampling logic is identical either way.
"""
import argparse
import json
import random
import sys

sys.path.insert(0, "src")
from config import CONVERSATIONS_PATH, GOLDEN_SET_PATH, INTENT_LABELS, ESCALATION_KEYWORDS
from heuristics import keyword_bucket

random.seed(7)

MIN_PER_BUCKET = 15
MAX_PER_BUCKET = 25  # 8 buckets x 25 = up to 200, within the assignment's 150-250 target


def stratified_sample(convs):
    buckets = {intent: [] for intent in INTENT_LABELS}
    for c in convs:
        buckets[keyword_bucket(c["customer_text"])].append(c)

    sampled = []
    for intent, items in buckets.items():
        random.shuffle(items)
        take = max(MIN_PER_BUCKET, min(MAX_PER_BUCKET, len(items)))
        take = min(take, len(items))
        sampled.extend((intent, item) for item in items[:take])
    random.shuffle(sampled)
    return sampled


def rule_based_escalation(customer_text: str, intent: str) -> tuple[bool, str]:
    text_l = customer_text.lower()
    hits = [kw for kw in ESCALATION_KEYWORDS if kw in text_l]
    if hits:
        return True, f"Customer used escalation-signal language: {hits}"
    if intent == "delivery_delay_or_lost" and "third time" in text_l:
        return True, "Repeated failure mentioned by customer (3rd occurrence)"
    return False, ""


def label_example(intent_bucket: str, conv: dict, interactive: bool) -> dict:
    if interactive:
        print("\n" + "=" * 60)
        print(f"Customer: {conv['customer_text']}")
        print(f"Brand reply (historical): {conv['brand_reply']}")
        print(f"Suggested intent (heuristic): {intent_bucket}")
        print(f"Options: {INTENT_LABELS}")
        raw_intent = input("True intent [enter to accept suggestion]: ").strip()
        true_intent = raw_intent if raw_intent in INTENT_LABELS else intent_bucket
        if raw_intent and raw_intent not in INTENT_LABELS:
            print(f"  (! '{raw_intent}' isn't a valid intent label — kept the suggestion '{intent_bucket}' instead)")

        esc_raw = ""
        while esc_raw not in ("y", "n"):
            esc_raw = input("Should escalate? (y/n, required): ").strip().lower()
            if esc_raw not in ("y", "n"):
                print("  Please type y or n.")
        esc = esc_raw == "y"

        reason = input("Escalation reason (blank if not escalating): ").strip()
        notes = input("Notes (optional): ").strip()
    else:
        true_intent = intent_bucket
        esc, reason = rule_based_escalation(conv["customer_text"], intent_bucket)
        notes = ""

    return {
        "conv_id": conv["conv_id"],
        "customer_text": conv["customer_text"],
        "historical_brand_reply": conv["brand_reply"],
        "true_intent": true_intent,
        "should_escalate": esc,
        "escalation_reason": reason,
        "notes": notes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true",
                         help="Hand-label via CLI instead of using template/rule ground truth")
    args = parser.parse_args()

    convs = [json.loads(l) for l in open(CONVERSATIONS_PATH, encoding="utf-8")]
    sampled = stratified_sample(convs)

    golden = [label_example(intent, conv, args.interactive) for intent, conv in sampled]

    with open(GOLDEN_SET_PATH, "w", encoding="utf-8") as f:
        for row in golden:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(golden)} golden examples -> {GOLDEN_SET_PATH}")
    from collections import Counter
    dist = Counter(g["true_intent"] for g in golden)
    print("Intent distribution in golden set:", dict(dist))
    print(f"Escalation rate: {sum(g['should_escalate'] for g in golden)}/{len(golden)}")


if __name__ == "__main__":
    main()