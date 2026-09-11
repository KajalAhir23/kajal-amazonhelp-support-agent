# Decision Log — 12 non-obvious decisions

1. **Built against a synthetic sample instead of the real Kaggle dataset.**
   Kaggle requires a personal login unavailable in the build environment.
   Chose to build a fully-functional pipeline against a schema-identical
   synthetic sample rather than block on data access, with a one-command
   swap-in path (`src/download_real_data.sh`) for the real thing. Cost:
   every metric in this repo is currently a plumbing demo, not a real
   result — disclosed explicitly in the report.

2. **Chose Groq over OpenAI/Anthropic for the LLM.** Free tier, no
   credit card, fast Llama models — removes cost/access as a blocker for
   whoever runs this repo next.

3. **Every LLM-dependent module has an offline fallback**
   (`llm_client.py` returns a mock; `classify_intent.py` falls back to
   keyword heuristic; `draft_reply.py` falls back to the closest
   retrieved historical reply; `llm_judge.py` falls back to a length/
   keyword heuristic). This means the entire pipeline is runnable and
   demoable with zero setup, at the cost of those numbers being
   non-representative until a key is added — a trade-off made explicit
   everywhere it applies rather than hidden.

4. **Retrieval defaults to TF-IDF, not sentence-transformer embeddings**,
   because the build sandbox can't reach huggingface.co to download
   model weights. `EmbeddingRetriever` is implemented and switchable via
   one env var for anyone running this with normal internet access.

5. **Golden-set sampling is stratified with a floor and ceiling per
   intent bucket** (15-45), not pure random sampling, so rare intents
   like cancellation aren't drowned out and no single intent dominates
   evaluation.

6. **Billing disputes are always escalated, regardless of confidence.**
   A deliberate policy choice: money-risk conversations shouldn't be
   autonomously handled by a first version of this agent, even when the
   drafted reply looks fine. This caps the auto-handle rate on purpose.

7. **Escalation combines rule signals first, LLM judgment second** (only
   called if no rule fires) rather than always asking the LLM. Cheaper,
   faster, and more auditable for a support team that needs to debug
   why a specific message escalated.

8. **The LLM-as-judge rubric has 3 separate dimensions** (grounded,
   correctness, tone) instead of one overall quality score, so failure
   analysis can pinpoint *why* a reply is bad, not just that it is.

9. **Judge-human agreement is tested on deliberately adversarial pairs**
   (same customer message, one good reply + one wrong/hallucinated
   reply) rather than random examples, because a judge that just
   rewards "sounds polite" would pass a random-example check while
   missing dangerous cases like an invented refund amount.

10. **Escalation recall is prioritized over precision in the design**
    (low threshold, hard-coded escalation intents) — a missed
    escalation (wrongly auto-replying) is worse for a real business
    than an unnecessary one, even though this currently hurts the
    reported precision number.

11. **Two baselines, not one**, as required: a trivial majority-class
    baseline AND a simple TF-IDF+LogisticRegression baseline, so the
    LLM classifier's value is judged against a real bar, not just "vs.
    nothing."

12. **Repo uses plain scripts (not notebooks) for everything except
    EDA.** Scripts are easier to run in the <15 minute reproduction
    window the assignment requires, are diffable in git history, and
    double as the evaluation harness rather than being throwaway
    exploration code.
