"""
Validates the LLM-as-judge against human ratings (data/golden/human_quality_ratings.jsonl).

The 10 examples are deliberately paired (good vs bad reply to the SAME
customer message) so the check isn't just "does the judge like generically
nice-sounding text" — it specifically tests whether the judge can tell a
correctly-grounded reply apart from a wrong/hallucinated/off-policy one.

NOTE: with no GROQ_API_KEY configured, llm_judge.py uses its offline
heuristic fallback, so the agreement numbers below reflect that heuristic,
not the real Groq judge. Add a free key (.env.example) and re-run this
script for the real evidence that goes in the report — the mechanism
(pairing + correlation + per-dimension breakdown) is identical either way.
"""
import json
import sys

from scipy.stats import pearsonr

sys.path.insert(0, "src")
from llm_judge import judge, overall_score
from config import GROQ_API_KEY


def main():
    rows = [json.loads(l) for l in open("data/golden/human_quality_ratings.jsonl", encoding="utf-8")]

    human_overall, judge_overall = [], []
    per_dim_human = {"grounded": [], "correctness": [], "tone": []}
    per_dim_judge = {"grounded": [], "correctness": [], "tone": []}

    print(f"{'id':6s} {'human':>7s} {'judge':>7s}  note")
    for row in rows:
        h_scores = row["human_scores"]
        h_overall = sum(h_scores.values()) / 3
        j_result = judge(row["customer_text"], row["candidate_reply"], row.get("reference_reply", ""))
        j_overall = overall_score(j_result)

        human_overall.append(h_overall)
        judge_overall.append(j_overall)
        for dim in per_dim_human:
            per_dim_human[dim].append(h_scores[dim])
            per_dim_judge[dim].append(j_result.get(dim, 3))

        print(f"{row['id']:6s} {h_overall:7.2f} {j_overall:7.2f}  {row['note'][:60]}")

    if len(set(human_overall)) > 1 and len(set(judge_overall)) > 1:
        r, p = pearsonr(human_overall, judge_overall)
    else:
        r, p = float("nan"), float("nan")

    print(f"\nOverall-score Pearson correlation (judge vs human): r={r:.3f} (p={p:.3f})")
    for dim in per_dim_human:
        if len(set(per_dim_human[dim])) > 1 and len(set(per_dim_judge[dim])) > 1:
            r_dim, _ = pearsonr(per_dim_human[dim], per_dim_judge[dim])
            print(f"  {dim:12s} r={r_dim:.3f}")

    if not GROQ_API_KEY:
        print("\n[!] Ran with the OFFLINE heuristic judge (no GROQ_API_KEY set). "
              "Add a free key and re-run for the real judge-agreement numbers.")


if __name__ == "__main__":
    main()
