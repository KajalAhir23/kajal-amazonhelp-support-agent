"""
Drafts a reply to a customer message, grounded in the top-k most similar
historically-resolved AmazonHelp conversations (src/retrieval.py).

This is deliberately RAG-style rather than pure generation: the LLM is
instructed to base its reply on the retrieved patterns (tone, what info
to ask for, what resolution to offer) rather than inventing policy.
"""
import sys

sys.path.insert(0, "src")
from llm_client import chat, has_real_llm
from retrieval import build_retriever

SYSTEM_PROMPT = """You are Kajal, an AI support agent replying on behalf of AmazonHelp on Twitter.
Rules:
- Keep replies under 280 characters, in AmazonHelp's typical tone (polite, brief, asks to DM for order-specific details).
- Base your reply on the RETRIEVED HISTORICAL REPLIES provided — match their pattern
  (what they apologize for, what they ask the customer to do next) rather than inventing new policy.
- Never make promises about compensation, refund amounts, or timelines the historical replies don't support.
- Do not include hashtags."""


def draft(customer_text: str, retriever=None, k: int = 3) -> dict:
    retriever = retriever or build_retriever()
    retrieved = retriever.top_k(customer_text, k=k)

    context_block = "\n".join(
        f'- Similar past issue: "{conv["customer_text"]}"\n'
        f'  How it was resolved: "{conv["brand_reply"]}"'
        for conv, _score in retrieved
    )

    prompt = (
        f"RETRIEVED HISTORICAL REPLIES:\n{context_block}\n\n"
        f'NEW CUSTOMER MESSAGE:\n"{customer_text}"\n\n'
        f"Draft the reply now:"
    )

    if has_real_llm():
        reply = chat(SYSTEM_PROMPT, prompt, temperature=0.4).strip()
    else:
        # Offline fallback: reuse the closest historical reply's *pattern*
        # directly, clearly marked, so the pipeline is demoable without a key.
        reply = retrieved[0][0]["brand_reply"] if retrieved else \
            "We're sorry for the trouble — please DM your order number so we can help. ^KJ"

    return {
        "customer_text": customer_text,
        "drafted_reply": reply,
        "grounding_examples": [c["conv_id"] for c, _ in retrieved],
        "grounding_scores": [round(s, 3) for _, s in retrieved],
    }


if __name__ == "__main__":
    result = draft("Hi @AmazonHelp my order #555222 was supposed to arrive yesterday, tracking still says processing")
    print(f"Drafted reply: {result['drafted_reply']}")
    print(f"Grounded in: {result['grounding_examples']} (scores: {result['grounding_scores']})")
