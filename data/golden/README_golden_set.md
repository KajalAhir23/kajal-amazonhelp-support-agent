# Golden Evaluation Set — Methodology

**Size:** 213 examples (within the required 150–250 range).

## Sampling
Conversation pairs were stratified by a cheap keyword heuristic
(`src/heuristics.py`) into the 8 intent buckets, then sampled with a
floor of 15 and ceiling of 45 per bucket. This avoids two failure modes
of pure random sampling: (a) common intents like refund/delivery
drowning out rare ones like cancellation, and (b) an evaluation set that
looks nothing like the real intent distribution because of arbitrary
floors alone — the ceiling caps dominant buckets so no single intent
exceeds ~20% of the set.

## Labelling
- **On the synthetic sample used for pipeline development:** ground-truth
  intent comes from the known generation template, and escalation is
  assigned by an explicit rule (angry/legal language, or "third time"
  repeated-failure phrasing → escalate). This is a stand-in for human
  labelling because the synthetic data's true label is known by
  construction, not because we consider heuristic labels equivalent to
  human judgment.
- **On the real Kaggle data:** run `python src/build_golden_set.py
  --interactive` — this walks a human labeller through each sampled
  example via the CLI (shows the tweet + historical brand reply, asks for
  true intent, escalation decision, and reason) and writes the same
  schema. This is the path we'd use for the actual submission once the
  real `twcs.csv` is downloaded locally.

## Schema
Each line in `golden_eval_set.jsonl`:
```json
{
  "conv_id": "...",
  "customer_text": "...",
  "historical_brand_reply": "...",
  "true_intent": "one of the 8 labels",
  "should_escalate": true/false,
  "escalation_reason": "...",
  "notes": "..."
}
```

## Known limitation
This is disclosed in the report's "what's misleading about my headline
number" section: labels generated from template/rule ground truth on
synthetic data are cleaner and easier than real human-labelled tweets,
which have typos, sarcasm, multi-intent messages, and ambiguity. Metrics
on this golden set should be read as an upper bound until re-run against
a human-labelled golden set from the real dataset.
