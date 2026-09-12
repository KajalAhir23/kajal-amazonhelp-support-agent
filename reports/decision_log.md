# Decision Log — 13 non-obvious decisions

1. **Initially built against a synthetic sample, then migrated to the
   real Kaggle dataset once available.** Kaggle requires a personal
   login unavailable in the original build environment, so the pipeline
   was first built end-to-end against a schema-identical synthetic
   sample to avoid blocking on data access. All final numbers in
   `reports/report.md` are from the REAL dataset (168,814 real AmazonHelp
   conversation pairs, subsampled to 20,000) — the synthetic sample now
   only exists as `src/make_synthetic_sample.py`, useful for anyone who
   wants to sanity-check the pipeline offline before setting up Kaggle
   access themselves.

2. **Chose Groq over OpenAI/Anthropic for the LLM.** Free tier, no
   credit card, fast models — removes cost/access as a blocker for
   whoever runs this repo next. Note: Groq deprecated `llama-3.3-70b-versatile`
   to Enterprise-only partway through this project; the repo now uses
   `openai/gpt-oss-120b`, documented in `.env.example`.

3. **Every LLM-dependent module has an offline fallback**
   (`llm_client.py` returns a mock; `classify_intent.py` falls back to
   keyword heuristic; `draft_reply.py` falls back to the closest
   retrieved historical reply; `llm_judge.py` falls back to a length/
   keyword heuristic). This means the pipeline is runnable and demoable
   with zero setup, and also degrades gracefully under free-tier rate
   limits (see decision 13) rather than hard-failing.

4. **Retrieval defaults to TF-IDF, not sentence-transformer embeddings**,
   because the original build sandbox couldn't reach huggingface.co to
   download model weights. `EmbeddingRetriever` is implemented and
   switchable via one env var (`RETRIEVAL_BACKEND=embedding`) for use
   with normal internet access.

5. **Golden-set sampling is stratified with a floor and ceiling per
   intent bucket** (15-25 on real data), not pure random sampling, so
   rare intents like cancellation aren't drowned out and no single
   intent dominates evaluation.

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
    than an unnecessary one, even though this can hurt the reported
    precision number.

11. **Two baselines, not one**, as required: a trivial majority-class
    baseline AND a simple TF-IDF+LogisticRegression baseline, so the
    LLM classifier's value is judged against a real bar, not just "vs.
    nothing." On real data this bar turned out to matter: the LLM only
    modestly beat TF-IDF (68.3% vs 66.7%), a genuinely useful finding
    that a single-baseline comparison would have hidden.

12. **Golden-set escalation labels are rule-based, not hand-judged, for
    all 200 examples — a decision made empirically, not by default.**
    Two earlier attempts at manually labelling escalation across 200
    examples in a row produced implausible rates (~1% then ~100%) from
    rapid, fatigue-driven y/n input. Rather than trust either pass, the
    same deterministic rule used in `src/escalate.py` was applied
    uniformly as ground truth instead — disclosed as a limitation in
    the report rather than presented as human judgment. Only intent
    labels (the part needing real human judgment) were hand-labelled
    for a subset (50 of 200), with the source tracked per-row via
    `intent_label_source`.

13. **Repo uses plain scripts (not notebooks) for everything except
    EDA**, and includes explicit rate-limit backoff/throttling in
    `llm_client.py` for the free-tier API. Scripts are easier to run in
    the <15 minute reproduction window the assignment requires and
    double as the evaluation harness; the throttling exists because
    real free-tier usage hit 429 rate-limit errors during development,
    a genuine constraint worth designing around rather than ignoring.
