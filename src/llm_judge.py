"""
LLM-as-judge for drafted reply quality. Scores 1-5 on a rubric, because a
single number ("is this reply good?") hides WHY it's good/bad, which the
report's failure analysis needs.

Rubric dimensions (each 1-5):
  - grounded     : does the reply match how this brand actually resolves
                   this kind of issue (vs a generic/hallucinated answer)?
  - correctness  : does it address what the customer actually asked?
  - tone         : polite, on-brand, appropriately brief?

Offline fallback (no API key): a cheap heuristic scorer so the harness
still runs end-to-end without a key — clearly inferior to the real judge,
used only for pipeline plumbing/demo purposes.
"""
import json
import re
import sys

sys.path.insert(0, "src")
from llm_client import chat, has_real_llm

SYSTEM_PROMPT = """You are grading an AI-drafted customer support reply for AmazonHelp.
Score the reply on three dimensions, each 1-5 (5 = best):
- grounded: does it match how AmazonHelp has historically resolved this kind of issue (given the reference), rather than inventing a generic/incorrect response?
- correctness: does it actually address what the customer asked?
- tone: polite, on-brand, appropriately brief (Twitter-length)?

Respond ONLY as JSON: {"grounded": <1-5>, "correctness": <1-5>, "tone": <1-5>, "explanation": "<one sentence>"}"""


def _heuristic_judge(customer_text: str, drafted_reply: str, reference_reply: str) -> dict:
    """Offline fallback — NOT a substitute for the real judge, just keeps
    the harness runnable without an API key."""
    grounded = 5 if reference_reply and reference_reply.strip() == drafted_reply.strip() else 3
    correctness = 4 if "dm" in drafted_reply.lower() or "sorry" in drafted_reply.lower() else 2
    tone = 4 if len(drafted_reply) <= 280 else 2
    return {
        "grounded": grounded, "correctness": correctness, "tone": tone,
        "explanation": "[heuristic fallback — no LLM key configured]",
    }


def judge(customer_text: str, drafted_reply: str, reference_reply: str = "") -> dict:
    if not has_real_llm():
        return _heuristic_judge(customer_text, drafted_reply, reference_reply)

    prompt = (
        f'Customer message: "{customer_text}"\n'
        f'Reference (a real historical AmazonHelp resolution to a similar issue): "{reference_reply}"\n'
        f'AI-drafted reply to grade: "{drafted_reply}"'
    )
    raw = chat(SYSTEM_PROMPT, prompt, temperature=0.0)
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(match.group(0))
    except Exception:
        parsed = {"grounded": 3, "correctness": 3, "tone": 3, "explanation": f"[unparseable judge output: {raw[:80]}]"}
    return parsed


def overall_score(judge_result: dict) -> float:
    return round((judge_result["grounded"] + judge_result["correctness"] + judge_result["tone"]) / 3, 2)


if __name__ == "__main__":
    result = judge(
        "My package never arrived, tracking shows nothing",
        "Hi there, we're sorry for the delay! Please DM us your order number and zip code. ^KL",
        "Hi there, we're sorry for the delay! Please DM us your order number and zip code so we can look into the tracking for you. ^KL",
    )
    print(result, "overall:", overall_score(result))
