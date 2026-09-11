"""
LLM-based (Groq, free tier) few-shot intent classifier — the actual
system used in the agent, evaluated against the two baselines in
src/baseline_classifiers.py.

Few-shot examples are pulled from the training split so the prompt
grounds the model in real labelled examples rather than relying purely
on the label names.
"""
import json
import re
import sys

sys.path.insert(0, "src")
from config import INTENT_LABELS
from llm_client import chat, has_real_llm
from heuristics import keyword_bucket

SYSTEM_PROMPT = f"""You are an intent classifier for AmazonHelp customer support tweets.
Classify the customer's message into EXACTLY ONE of these intents:
{json.dumps(INTENT_LABELS, indent=2)}

Respond with ONLY the intent label, nothing else."""


def build_few_shot_block(train_examples: list, k_per_intent: int = 2) -> str:
    by_intent = {}
    for ex in train_examples:
        by_intent.setdefault(ex["true_intent"], []).append(ex)
    blocks = []
    for intent in INTENT_LABELS:
        for ex in by_intent.get(intent, [])[:k_per_intent]:
            blocks.append(f'Message: "{ex["customer_text"]}"\nIntent: {intent}')
    return "\n\n".join(blocks)


def classify(customer_text: str, few_shot_block: str = "") -> str:
    prompt = ""
    if few_shot_block:
        prompt += f"Examples:\n{few_shot_block}\n\n"
    prompt += f'Now classify this message:\nMessage: "{customer_text}"\nIntent:'

    raw = chat(SYSTEM_PROMPT, prompt, temperature=0.0)

    if not has_real_llm():
        # No key configured: fall back to the keyword heuristic so the
        # pipeline is still fully runnable and demoable end-to-end.
        return keyword_bucket(customer_text)

    # Sanitize: LLM should return just the label, but guard against
    # extra text/punctuation.
    cleaned = re.sub(r"[^a-z_]", "", raw.strip().lower())
    for label in INTENT_LABELS:
        if label in cleaned:
            return label
    return "general_inquiry_or_other"  # safe fallback if LLM output is unparseable


if __name__ == "__main__":
    # Quick smoke test
    test_msgs = [
        "My order still hasn't arrived and tracking shows nothing for 5 days",
        "I can't log into my account no matter what I try",
    ]
    for msg in test_msgs:
        print(f"{msg[:50]:50s} -> {classify(msg)}")
