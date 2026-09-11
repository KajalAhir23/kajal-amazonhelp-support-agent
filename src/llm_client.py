"""
Thin wrapper around the Groq API (free tier) used by the classifier,
reply drafter, and LLM-as-judge. Centralized here so:
  - there's one place that handles "no API key configured yet" gracefully
    (falls back to a deterministic mock so the rest of the pipeline is
    still runnable/demoable without signing up first),
  - rate limits (free tier caps requests/min) are retried with backoff
    instead of crashing the whole eval run, and
  - swapping providers/models later only touches this file.
"""
import sys
import time

sys.path.insert(0, "src")
from config import GROQ_API_KEY, GROQ_MODEL

_client = None
_warned = False

MIN_DELAY_BETWEEN_CALLS = 1.2  # seconds - keeps us under free-tier RPM caps
_last_call_time = 0.0


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
    mock string if no GROQ_API_KEY is set. Retries with exponential
    backoff on 429 (rate limit) errors, which the free tier hits easily
    when running many calls in a row (e.g. classifying a whole test set)."""
    global _warned, _last_call_time
    client = _get_client()
    if client is None:
        if not _warned:
            print("[llm_client] No GROQ_API_KEY set — using MOCK LLM responses. "
                  "Add a free key to .env to get real outputs (see .env.example).")
            _warned = True
        return "[MOCK_LLM_RESPONSE] " + user_prompt[:60]

    # Simple client-side throttle so we don't even trigger 429s in the first place.
    elapsed = time.time() - _last_call_time
    if elapsed < MIN_DELAY_BETWEEN_CALLS:
        time.sleep(MIN_DELAY_BETWEEN_CALLS - elapsed)

    max_retries = 5
    backoff = 3.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model or GROQ_MODEL,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            _last_call_time = time.time()
            return resp.choices[0].message.content
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
            if is_rate_limit and attempt < max_retries - 1:
                print(f"[llm_client] Rate limited, waiting {backoff:.0f}s before retry "
                      f"({attempt + 1}/{max_retries})...")
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
