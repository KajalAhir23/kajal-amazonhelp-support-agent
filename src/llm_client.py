"""
Thin wrapper around the Groq API (free tier) used by the classifier,
reply drafter, and LLM-as-judge. Centralized here so:
  - there's one place that handles "no API key configured yet" gracefully
    (falls back to a deterministic mock so the rest of the pipeline is
    still runnable/demoable without signing up first), and
  - swapping providers later only touches this file.
"""
import sys
sys.path.insert(0, "src")
from config import GROQ_API_KEY, GROQ_MODEL

_client = None
_warned = False


def _get_client():
    global _client
    if _client is None and GROQ_API_KEY:
        from groq import Groq
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def has_real_llm() -> bool:
    return bool(GROQ_API_KEY)


def chat(system_prompt: str, user_prompt: str, model: str = None, temperature: float = 0.2) -> str:
    """Returns the model's text response. Falls back to a clearly-marked
    mock string if no GROQ_API_KEY is set, so downstream code (and this
    repo's tests) still run end-to-end without a key."""
    global _warned
    client = _get_client()
    if client is None:
        if not _warned:
            print("[llm_client] No GROQ_API_KEY set — using MOCK LLM responses. "
                  "Add a free key to .env to get real outputs (see .env.example).")
            _warned = True
        return "[MOCK_LLM_RESPONSE] " + user_prompt[:60]

    resp = client.chat.completions.create(
        model=model or GROQ_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content
