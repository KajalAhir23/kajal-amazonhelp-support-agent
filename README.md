# Kajal — AI Support Agent for AmazonHelp

An AI agent that reads a customer's tweet to @AmazonHelp, classifies its
intent, drafts a reply grounded in how AmazonHelp has historically
resolved similar issues, and decides whether to auto-handle or escalate
to a human — with a stated reason.

Built for the Hiver SDE Intern take-home assignment.

**Read `reports/report.md` first**, especially §4 ("what's misleading
about my headline number") — this repo currently runs against a
synthetic sample (schema-identical to the real Kaggle data) because the
real dataset needs a personal Kaggle login. Every script below produces
real results with zero code changes once the real data + a free API key
are added (2 commands, see below).

## Quickstart (reproduces headline results in under 15 minutes)

```bash
pip install -r requirements.txt

# Optional but recommended — free Groq key for real LLM outputs.
# Without this, every script still runs end-to-end using documented
# offline fallbacks (see reports/decision_log.md, item 3).
cp .env.example .env   # then paste your key from console.groq.com/keys

# 1. Generate data (synthetic sample; swap for real data below if you want)
python src/make_synthetic_sample.py
python src/data_prep.py

# 2. EDA + intent taxonomy justification
python notebooks/01_eda.py

# 3. Build the golden evaluation set (213 examples)
python src/build_golden_set.py

# 4. Baseline comparison (trivial vs simple vs LLM classifier)
python src/compare_baselines.py

# 5. Full evaluation harness (intent + escalation + reply-quality judge)
python src/run_eval.py --limit 30

# 6. Judge-human agreement validation
python src/judge_agreement.py

# 7. Try the agent interactively
python src/run_pipeline.py --message "my order never arrived, its been a week"
python src/run_pipeline.py --n 5   # or run on random golden-set examples
```

### To run against the REAL Kaggle dataset instead
```bash
bash src/download_real_data.sh   # needs your own ~/.kaggle/kaggle.json
python src/data_prep.py          # auto-prefers twcs.csv over the synthetic sample
python src/build_golden_set.py --interactive   # hand-label via CLI instead of template ground truth
```

## Project structure
```
src/
  config.py                 central config (paths, intent taxonomy, thresholds)
  make_synthetic_sample.py  generates a schema-identical synthetic dataset
  download_real_data.sh     downloads the real Kaggle twcs.csv (needs your login)
  data_prep.py              filters to AmazonHelp, pairs customer<->brand tweets
  heuristics.py             cheap keyword bucket used only for EDA/sampling
  build_golden_set.py       stratified sampling + CLI hand-labelling tool
  baseline_classifiers.py   trivial + TF-IDF/LogReg baselines
  llm_client.py             Groq API wrapper with offline mock fallback
  classify_intent.py        few-shot LLM intent classifier
  compare_baselines.py      3-way classifier comparison on held-out split
  retrieval.py              TF-IDF / embedding retriever for grounding
  draft_reply.py            retrieval-grounded reply drafting
  escalate.py               rule + LLM escalation decision, always states a reason
  metrics.py                intent + escalation automated metrics
  llm_judge.py              LLM-as-judge rubric for reply quality
  judge_agreement.py        judge vs. hand-labelled human quality ratings
  run_eval.py               end-to-end evaluation harness over the golden set
  run_pipeline.py           interactive CLI: classify -> draft -> escalate
notebooks/01_eda.py         EDA justifying the intent taxonomy
data/raw/                   synthetic sample (real twcs.csv is gitignored)
data/processed/             cleaned conversation pairs
data/golden/                golden eval set + hand-labelled judge-agreement set
reports/                    report.md, decision_log.md, failure_analysis.md
```

## Deliverables checklist
- [x] Runnable pipeline, reproducible in <15 min (`Quickstart` above)
- [x] Golden evaluation set — 213 examples (`data/golden/golden_eval_set.jsonl`, methodology in `data/golden/README_golden_set.md`)
- [x] Evaluation harness — automated metrics + LLM-as-judge + judge-human agreement (`src/run_eval.py`, `src/judge_agreement.py`)
- [x] Report — `reports/report.md` (problem framing, baselines, failure analysis, misleading-number section, next steps)
- [x] Decision log — `reports/decision_log.md` (12 decisions)
