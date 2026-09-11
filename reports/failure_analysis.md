# Failure Analysis — Top 5 Failure Modes

All examples below are real outputs from this repo's own eval runs
(`src/run_eval.py`, `src/compare_baselines.py`, `src/judge_agreement.py`),
not hypothetical.

## 1. Perfect classifier scores are an artifact of synthetic data, not real skill
**Observed:** Both the TF-IDF+LogReg baseline AND the LLM classifier hit
100% accuracy / macro-F1 on the golden set.
**Example:** Every "delivery_delay_or_lost" example in the synthetic
sample contains near-identical template phrasing ("was supposed to
arrive", "tracking hasn't updated"), making the classes trivially linearly
separable.
**Hypothesis:** Real AmazonHelp tweets have typos, sarcasm, multiple
intents in one tweet, and much more lexical variety. This 100% number
will drop substantially on the real Kaggle data — it's flagged explicitly
in the report's "misleading headline number" section rather than
presented as a real result.

## 2. Escalation over-triggers in the fully-offline configuration
**Observed:** On a 30-example run, escalation precision was 0.556 while
recall was 1.000 (`run_eval.py` output).
**Example:** Every message got flagged whenever retrieval confidence fell
below the 0.55 threshold — but the offline TF-IDF retriever's confidence
scores are compressed into a narrower range than semantic embeddings
would produce, so the threshold tuned with embeddings in mind was too
aggressive for TF-IDF similarity scores.
**Hypothesis:** The `LOW_CONFIDENCE_THRESHOLD` needs separate calibration
per retrieval backend, or better, needs to be a percentile-based cutoff
computed from the corpus's own similarity-score distribution rather than
a fixed constant.

## 3. LLM-as-judge (offline heuristic fallback) has weak tone discrimination
**Observed:** In `judge_agreement.py`, the "tone" dimension showed zero
variance across all 10 hand-labelled examples (the heuristic just checks
reply length), so its correlation with human tone ratings couldn't even
be computed.
**Example:** A generic non-answer ("Thanks for reaching out! Have a great
day!") scored the same tone as a correctly-grounded, on-brand reply,
because both are short.
**Hypothesis:** This is a known, disclosed limitation of the *offline
fallback* judge specifically — it exists only so the harness runs without
an API key. The real Groq-based judge (once a key is added) evaluates
tone via the LLM's own judgment, not a length heuristic, and should show
real variance. This must be re-validated before trusting judge scores.

## 4. Retrieval returns identical scores for near-duplicate synthetic examples
**Observed:** In `retrieval.py`'s smoke test, the top-3 retrieved examples
for a delivery-delay query all scored exactly 0.328 — because the
synthetic generator reuses the same 10 templates with only order-number
substitutions.
**Example:** Retrieval can't distinguish "arrived 3 days late" from
"arrived 3 days late" (order #A) vs (order #B) — there's no real semantic
diversity to rank.
**Hypothesis:** Retrieval quality is currently untestable in a meaningful
way on synthetic data. This needs re-evaluation on the real dataset where
genuine paraphrase diversity exists, ideally with the embedding backend
(`RETRIEVAL_BACKEND=embedding`) rather than TF-IDF, since embeddings
should better separate near-duplicate vs genuinely-different complaints.

## 5. Billing disputes are always escalated — by design, but this caps recall on "safe" auto-handling
**Observed:** `escalate.py`'s hard rule escalates 100% of
`billing_or_charge_dispute` messages regardless of confidence.
**Example:** Even a simple, unambiguous double-charge question with a
high-confidence grounded reply gets escalated.
**Hypothesis:** This is a deliberate, disclosed policy choice (money-risk
intents shouldn't be auto-handled at all in an early version of this
agent — see decision log), not a bug. But it means the system's
auto-handle rate is capped below what raw classification accuracy alone
would suggest, which is exactly the kind of thing the "misleading
headline number" section needs to call out.
