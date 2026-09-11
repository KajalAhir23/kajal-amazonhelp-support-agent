# Report — Kajal: AI Support Agent for AmazonHelp

## 1. Problem framing

For a brand like AmazonHelp on Twitter, "good" doesn't mean "sounds
plausible" — it means: **matches how AmazonHelp has actually resolved
this exact kind of issue before**, asks for the right follow-up info,
never invents policy (refund amounts, timelines, compensation) that
isn't supported by precedent, and correctly recognizes when a human
should take over instead of the bot.

**What I chose not to build:**
- **Full multi-turn dialogue management.** The agent handles the
  first customer message in a thread and drafts one reply; it doesn't
  manage a back-and-forth conversation state machine. Real support
  threads on Twitter are often 2-4 tweets deep, but the assignment's
  core ask (classify, draft, escalate) is about the decision at each
  turn, not conversation orchestration — extending to multi-turn is a
  "what's next" item, not core scope.
- **Fine-tuning a model.** With a small hand-labelled golden set and a
  free-tier LLM API, few-shot prompting is more sample-efficient and
  faster to iterate on than fine-tuning, and is fully reversible/auditable.
- **A learned escalation classifier.** Escalation decisions here combine
  explicit rules + an LLM policy check rather than a trained classifier,
  because for a first version of a system handling real customer trust
  and money, an auditable/debuggable rule set beats an opaque model —
  see decision log.

## 2. Results vs. baselines

Run via `python src/compare_baselines.py` (intent classification, 64-example
held-out test split of the golden set):

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TrivialBaseline (always predicts majority class) | 0.203 | 0.042 |
| SimpleBaseline (TF-IDF + Logistic Regression) | 1.000 | 1.000 |
| LLM Classifier (Groq, few-shot) | 1.000 | 1.000 |

Escalation decision (30-example subset, `python src/run_eval.py --limit 30`):
precision 0.556, recall 1.000, F1 0.714.

Reply quality (LLM-as-judge, 1-5 scale, offline heuristic fallback shown
here — see §5 for why this number needs re-validation): avg 4.29/5.

**Read §5 before trusting the accuracy numbers above.**

## 3. Failure analysis

See `reports/failure_analysis.md` for the full top-5 writeup with real
examples pulled directly from these eval runs. Summary:
1. Perfect classifier scores are a synthetic-data artifact, not real skill.
2. Escalation over-triggers (low precision) because the confidence
   threshold was tuned assuming embeddings, not TF-IDF similarity.
3. The offline judge fallback can't discriminate tone at all (zero variance).
4. Retrieval can't be meaningfully evaluated on near-duplicate synthetic data.
5. Billing disputes are always escalated by design, capping auto-handle rate.

## 4. What's misleading about my headline number (mandatory section)

**The 100% classifier accuracy and 4.29/5 reply quality are both
misleadingly good, for the same root cause: this repo runs against a
synthetic dataset generated from 10 fixed templates**, because the
real Kaggle dataset requires a personal login that isn't available in
the environment I built this in.

Specifically:
- The synthetic templates make each intent class near-perfectly
  linearly separable by keywords alone — real tweets have typos,
  sarcasm, multi-intent messages, and much messier phrasing. Expect
  real accuracy meaningfully below 100%.
- The reply-quality judge score of 4.29 is partly circular in offline
  mode: the fallback reply generator literally returns the closest
  retrieved historical reply verbatim when no LLM key is set, so it
  trivially "matches" the reference. The 5-point corroborating human
  vs. judge agreement check in `judge_agreement.py` (r=0.61 on the
  overall score, using the SAME offline heuristic judge) shows this
  fallback judge already has real discrimination problems on adversarial
  pairs, so 4.29 should not be read as "the agent's replies are 86% as
  good as ideal" — it's closer to "the plumbing works."
- The escalation precision of 0.556 (from real rule logic, not a
  fallback) is the most trustworthy number in this report, since the
  rule signals fire the same way regardless of data source.

**Before this could be trusted as a real evaluation:** the real
`twcs.csv` needs to be downloaded (`src/download_real_data.sh`), a
`GROQ_API_KEY` needs to be added (free, see `.env.example`), and
`build_golden_set.py --interactive` needs to be re-run for a genuinely
human-labelled golden set — at which point every script above produces
real numbers with zero code changes.

## 5. What I'd do next with one more week

1. Get the real Kaggle dataset + a Groq key running, and regenerate
   every number in this report against real data (this is the single
   highest-value next step — everything above is currently a
   plumbing/methodology demo, not a real evaluation).
2. Re-calibrate the escalation confidence threshold per retrieval
   backend using a percentile cutoff from the corpus's own score
   distribution, rather than one fixed constant.
3. Switch retrieval to the embedding backend (already wired in,
   `RETRIEVAL_BACKEND=embedding`) and re-run the retrieval quality
   check — TF-IDF can't meaningfully rank near-duplicate complaints.
4. Expand the golden set's escalation labels with a second human
   labeller and report inter-annotator agreement, since "should this
   escalate" is judgment-heavy and a single labeller's rule-based
   ground truth (used here) is a weak proxy for real ambiguity.
5. Add basic multi-turn support: track whether a customer has already
   DM'd order details in a previous tweet in the thread, so the agent
   doesn't re-ask for info it already has.
