# Golden Evaluation Set — Methodology

**Size:** 200 examples (within the required 150-250 range), built from
the REAL Kaggle "Customer Support on Twitter" dataset, filtered to
AmazonHelp, subsampled to 20,000 conversation pairs (`src/subsample_conversations.py`)
per the assignment's explicit allowance for subsampling.

## Sampling
Conversation pairs were stratified by a cheap keyword heuristic
(`src/heuristics.py`) into the 8 intent buckets, then sampled with a
floor of 15 and ceiling of 25 per bucket, giving an even 25-per-intent
final set.

## Labelling — hybrid approach (documented honestly)
- **50 examples: hand-labelled** by a human via
  `python src/build_golden_set.py --interactive --hand-label-n 50`,
  reviewing the actual tweet + AmazonHelp's real historical reply and
  confirming/correcting the intent label.
- **150 examples: heuristic-only**, using the keyword-bucket suggestion
  unreviewed. This is a disclosed limitation, not presented as human
  ground truth — flagged per-row via the `intent_label_source` field
  (`"human"` vs `"heuristic_unreviewed"`).
- **should_escalate is rule-based for all 200 examples**, using the same
  deterministic logic as the production `src/escalate.py` (anger/legal
  language, explicit repeated-failure mentions, or a money-risk intent
  like billing disputes). This was a deliberate choice made after two
  earlier fully-manual labelling passes produced implausible escalation
  rates (~1% and ~100%) from rapid, fatigue-driven y/n input across 200
  examples in a row — a consistent, auditable rule is more trustworthy
  ground truth here than a tired human reflexively hitting the same key.
  The final escalation rate (35/200, 17.5%) is in a plausible range for
  real support traffic.

## Schema
Each line in `golden_eval_set.jsonl`:
```json
{
  "conv_id": "...",
  "customer_text": "...",
  "historical_brand_reply": "...",
  "true_intent": "one of the 8 labels",
  "intent_label_source": "human | heuristic_unreviewed",
  "should_escalate": true/false,
  "escalation_reason": "...",
  "notes": "..."
}
```

## Known limitations (for the report's "misleading headline number" section)
1. 75% of intent labels are unreviewed heuristic output, not human
   judgment — any classifier metric computed against the full 200
   should be read alongside a metric computed against just the 50
   `intent_label_source == "human"` rows, which is the more trustworthy
   subset.
2. Escalation ground truth is rule-based, not human-judged — it tests
   whether the agent's rule engine matches a reference rule engine, not
   whether a human would actually agree these are the right calls to
   escalate. Real human-judged escalation labels are a "what's next"
   item.
