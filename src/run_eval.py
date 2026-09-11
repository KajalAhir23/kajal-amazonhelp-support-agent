"""
End-to-end evaluation harness: runs the full agent (classify -> draft ->
escalate) over the golden set and reports automated metrics + judge scores.

Usage: python src/run_eval.py [--limit N]
"""
import argparse
import json
import sys
import time

sys.path.insert(0, "src")
from config import GOLDEN_SET_PATH
from classify_intent import classify, build_few_shot_block
from retrieval import build_retriever
from draft_reply import draft
from escalate import decide
from llm_judge import judge, overall_score
from metrics import intent_metrics, escalation_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N golden examples (for speed)")
    args = parser.parse_args()

    golden = [json.loads(l) for l in open(GOLDEN_SET_PATH, encoding="utf-8")]
    if args.limit:
        golden = golden[: args.limit]

    few_shot_block = build_few_shot_block(golden, k_per_intent=1)
    retriever = build_retriever()

    pred_intents, true_intents = [], []
    pred_escalate, true_escalate = [], []
    judge_scores = []

    start = time.time()
    for i, row in enumerate(golden):
        elapsed = time.time() - start
        print(f"[{i+1}/{len(golden)}] ({elapsed:.0f}s elapsed) processing...", flush=True)

        pred_intent = classify(row["customer_text"], few_shot_block)
        pred_intents.append(pred_intent)
        true_intents.append(row["true_intent"])

        draft_result = draft(row["customer_text"], retriever=retriever)
        retrieval_conf = draft_result["grounding_scores"][0] if draft_result["grounding_scores"] else 0.0

        esc_result = decide(row["customer_text"], pred_intent, draft_result["drafted_reply"], retrieval_conf)
        pred_escalate.append(esc_result["escalate"])
        true_escalate.append(row["should_escalate"])

        j = judge(row["customer_text"], draft_result["drafted_reply"], row["historical_brand_reply"])
        judge_scores.append(overall_score(j))

    print("=" * 60)
    print(f"Evaluated {len(golden)} golden examples in {time.time()-start:.0f}s\n")

    im = intent_metrics(true_intents, pred_intents)
    print(f"Intent classification: accuracy={im['accuracy']:.3f}  macro-F1={im['macro_f1']:.3f}")

    em = escalation_metrics(true_escalate, pred_escalate)
    print(f"Escalation decision:   precision={em['precision']:.3f}  recall={em['recall']:.3f}  f1={em['f1']:.3f}")

    avg_judge = sum(judge_scores) / len(judge_scores)
    print(f"Reply quality (LLM judge, 1-5 avg): {avg_judge:.2f}")
    print("=" * 60)

    return {
        "n": len(golden),
        "intent_metrics": im,
        "escalation_metrics": em,
        "avg_judge_score": avg_judge,
    }


if __name__ == "__main__":
    main()
