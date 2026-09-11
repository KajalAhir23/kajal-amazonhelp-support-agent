"""
Decides whether a message should be auto-handled or escalated to a human,
with a stated reason. Combines cheap deterministic signals with an LLM
final call, because pure LLM escalation decisions are (a) more expensive
per message and (b) less auditable than rule signals for a support team
that needs to trust/debug the agent.

Signals considered (all stated in the reason string, not hidden):
  1. Escalation keyword hit (anger/legal/repeated-failure language)
  2. Low retrieval grounding confidence (the agent hasn't seen anything
     similar before -> shouldn't autonomously reply)
  3. Certain intents are hard-escalated regardless of confidence
     (billing disputes with money at risk -> always human-reviewed,
     a deliberate, disclosed design choice — see decision log)
"""
import sys

sys.path.insert(0, "src")
from config import LOW_CONFIDENCE_THRESHOLD, ESCALATION_KEYWORDS
from llm_client import chat, has_real_llm

ALWAYS_ESCALATE_INTENTS = {"billing_or_charge_dispute"}

SYSTEM_PROMPT = """You are a support-ops policy checker. Given a customer message,
its intent, a drafted reply, and confidence signals, decide if the drafted reply
is safe to send automatically or if it should go to a human agent instead.
Respond in the format:
DECISION: AUTO or ESCALATE
REASON: <one short sentence>"""


def rule_signals(customer_text: str, intent: str, retrieval_confidence: float) -> list:
    signals = []
    text_l = customer_text.lower()

    hits = [kw for kw in ESCALATION_KEYWORDS if kw in text_l]
    if hits:
        signals.append(f"escalation language detected ({hits})")

    if retrieval_confidence < LOW_CONFIDENCE_THRESHOLD:
        signals.append(
            f"low grounding confidence ({retrieval_confidence:.2f} < {LOW_CONFIDENCE_THRESHOLD}) "
            f"— no sufficiently similar past resolution found"
        )

    if intent in ALWAYS_ESCALATE_INTENTS:
        signals.append(f"intent '{intent}' is policy-flagged as always-human-reviewed (money at risk)")

    return signals


def decide(customer_text: str, intent: str, drafted_reply: str, retrieval_confidence: float) -> dict:
    signals = rule_signals(customer_text, intent, retrieval_confidence)

    if signals:
        # Deterministic path: any hard rule signal escalates immediately.
        # Cheaper and more auditable than always asking the LLM.
        return {
            "escalate": True,
            "reason": "; ".join(signals),
            "decision_path": "rule-based",
        }

    if not has_real_llm():
        return {
            "escalate": False,
            "reason": "No rule signals triggered; offline mode defaults to auto-handle "
                      "(no LLM key configured — see .env.example).",
            "decision_path": "offline-fallback",
        }

    prompt = (
        f'Customer message: "{customer_text}"\n'
        f"Classified intent: {intent}\n"
        f'Drafted reply: "{drafted_reply}"\n'
        f"Retrieval grounding confidence: {retrieval_confidence:.2f}\n"
        f"No rule-based red flags were triggered. Confirm this is safe to auto-send."
    )
    raw = chat(SYSTEM_PROMPT, prompt, temperature=0.0)

    escalate = "ESCALATE" in raw.upper()
    reason_line = next((l for l in raw.splitlines() if l.upper().startswith("REASON")), "REASON: (unspecified)")
    reason = reason_line.split(":", 1)[-1].strip()

    return {"escalate": escalate, "reason": reason, "decision_path": "llm-reviewed"}


if __name__ == "__main__":
    cases = [
        ("This is the third time my package has been delayed, unacceptable, I want a manager!",
         "delivery_delay_or_lost", "Sorry for the delay, please DM your order number.", 0.85),
        ("Where's my refund for order 12345", "refund_or_return_request",
         "We're sorry — please DM your order number.", 0.8),
        ("I was charged twice for the same item", "billing_or_charge_dispute",
         "Please DM your order and card last 4 digits.", 0.9),
    ]
    for text, intent, reply, conf in cases:
        result = decide(text, intent, reply, conf)
        print(f"[{'ESCALATE' if result['escalate'] else 'AUTO':8s}] ({result['decision_path']}) {result['reason']}")
