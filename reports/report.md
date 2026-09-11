# Report — Kajal: AI Support Agent for AmazonHelp

## 1. Problem framing

For a brand like AmazonHelp on Twitter, "good" doesn't mean "sounds
plausible" — it means: **matches how AmazonHelp has actually resolved
this exact kind of issue before**, asks for the right follow-up info,
never invents policy (refund amounts, timelines, compensation) that
isn't supported by precedent, and correctly recognizes when a human
should take over instead of the bot.

**What I chose not to build:**
- **Full multi-turn dialogue management.** The agent handles the first
  customer message in a thread and drafts one reply; it doesn't manage
  a back-and-forth conversation state machine.
- **Fine-tuning a model.** With a small hand-labelled set and a free-tier
  LLM API, few-shot prompting is more sample-efficient and faster to
  iterate on than fine-tuning, and is fully reversible/auditable.
- **A learned escalation classifier.** Escalation combines explicit rules
  + policy checks rather than a trained model, because for a system
  handling real customer trust and money, an auditable rule set beats
  an opaque model in a first version.

## 2. Results vs. baselines

Run against the REAL AmazonHelp subset of the Kaggle "Customer Support
on Twitter" dataset (168,814 real conversation pairs, subsampled to
20,000 for retrieval; golden set of 200 examples, 50 hand-labelled + 150
heuristic-labelled — see `data/golden/README_golden_set.md`).

Intent classification (`python src/compare_baselines.py`, 60-example
held-out test split):

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TrivialBaseline (always predicts majority class) | 0.117 | 0.026 |
| SimpleBaseline (TF-IDF + Logistic Regression) | 0.667 | 0.659 |
| LLM Classifier (Groq, gpt-oss-120b, few-shot) | 0.683 | 0.673 |

Full pipeline evaluation on 10 golden examples (`python src/run_eval.py --limit 10`):
- Intent classification: accuracy 0.800, macro-F1 0.829
- Escalation decision: precision 1.000, recall 1.000, F1 1.000 (small
  sample — see §4 on why this isn't fully trustworthy yet)
- Reply quality (real Groq LLM-as-judge, 1-5 scale): 4.73 average

Judge-human agreement (`python src/judge_agreement.py`, 10 adversarial
good/bad reply pairs, real Groq judge): overall Pearson r=0.724 (p=0.018),
with per-dimension correlation of r=0.846 (correctness), r=0.655
(grounded), r=0.422 (tone). The judge reliably flags wrong/hallucinated
replies but is systematically STRICTER than the human rater on replies
that are actually good (e.g. three examples the human rated 5/5 for
being correctly grounded were rated 1.67-2.67 by the judge) — see §4.

## 3. Failure analysis

See `reports/failure_analysis.md`. Key real findings from this run:
1. The LLM classifier only modestly beats TF-IDF+LogReg on real data
   (68.3% vs 66.7%) — a much smaller gap than expected, and a useful
   finding: for this task, simple retrieval-based features may already
   capture most of the signal a general-purpose LLM few-shot prompt
   extracts.
2. The real LLM judge is conservative/strict relative to a human on
   good replies, even while correctly penalizing bad ones — meaning
   raw judge scores likely UNDERSTATE reply quality, not overstate it.
3. Escalation precision/recall of 1.0 was measured on only 10 examples —
   not enough to trust yet; needs re-running on the full 200-example
   golden set before reporting as a real number.

## 4. What's misleading about my headline number (mandatory section)

- **The 4.73/5 average judge score and 1.0/1.0 escalation precision/recall
  were both measured on only 10 examples** (`--limit 10`, used to fit in
  the free-tier rate limit within a submission deadline). Neither is a
  reliable estimate yet — both need re-running against the full 200-example
  golden set for a real number worth trusting.
- **75% of the golden set's intent labels are unreviewed heuristic
  output, not human judgment** (`intent_label_source: "heuristic_unreviewed"`
  on 150/200 rows — see `data/golden/README_golden_set.md`). Any accuracy
  number computed against the full 200 should be treated as optimistic
  until re-computed against just the 50 hand-labelled rows.
- **Escalation ground truth is rule-based, not human-judged**, for a
  disclosed, deliberate reason: two earlier attempts at fully manual
  escalation labelling across 200 examples in a row produced implausible
  rates (~1% and ~100%) from rapid, fatigue-driven y/n input, which would
  have been worse ground truth than a consistent rule. This means the
  escalation metrics currently test "does the agent's rule match a
  reference rule," not "would a human agree these are the right calls."
- **The judge itself appears to be a strict, not lenient, grader** based
  on the 10-example agreement check — meaning real reply quality may be
  somewhat BETTER than the 4.73 average suggests, the opposite direction
  of the usual "LLM judges are sycophantic" concern. This needs a larger
  human-rated sample to confirm.

## 5. What I'd do next with one more week

1. Re-run `run_eval.py` and `judge_agreement.py` against the FULL
   200-example golden set (not `--limit 10`), spacing calls to respect
   the free-tier rate limit overnight if needed, for real trustworthy
   numbers on escalation and judge agreement.
2. Hand-label the remaining 150 heuristic-only intent rows properly, or
   at minimum report classifier accuracy separately on just the 50
   verified-human rows vs. the full 200.
3. Get a second human labeller to hand-judge escalation on a subsample,
   to actually validate the rule-based escalation ground truth against
   real human agreement rather than just internal consistency.
4. Switch retrieval to the embedding backend (`RETRIEVAL_BACKEND=embedding`)
   and compare retrieval quality against the current TF-IDF default on
   real (non-duplicate) data.
5. Investigate WHY the LLM classifier only modestly beat TF-IDF (68.3%
   vs 66.7%) — try more few-shot examples per intent, or a larger Groq
   model, to see if the gap widens with a stronger setup.
