# Kajal - AI Support Agent for AmazonHelp

An AI agent that reads a customer's tweet to @AmazonHelp, classifies its
intent, drafts a reply grounded in how AmazonHelp has historically
resolved similar issues, and decides whether to auto-handle or escalate
to a human - with a stated reason.

Built for the Hiver SDE Intern take-home assignment.

**Read `reports/report.md` first**, especially section 4 ("what's misleading
about my headline number") - all headline numbers come from the REAL
Kaggle dataset, but with disclosed limitations (small eval sample size,
partially heuristic-labelled golden set, rule-based escalation ground
truth). See `reports/decision_log.md` for why.

## Optional: Interactive Web UI

A simple Streamlit UI sits on top of the CLI pipeline, purely for demo
purposes (not required by the assignment, which only asks for a
runnable script-based pipeline):

```bash
pip install streamlit   # already in requirements.txt
python -m streamlit run app.py
```

Opens a browser page where you can type a customer message (or click a
sample from the golden set) and see the predicted intent, drafted reply,
which historical conversations it was grounded in, and the
auto-handle/escalate decision with its reason.

## Quickstart (reproduces headline results in under 15 minutes)

```bash
pip install -r requirements.txt

# Free Groq key for real LLM outputs (get one at console.groq.com/keys).
# Without this, every script still runs end-to-end using documented
# offline fallbacks (see reports/decision_log.md, item 3).
cp .env.example .env   # then paste your key

# 1. Get data - either the real Kaggle dataset (recommended) or the
#    synthetic sample (works fully offline, no Kaggle login needed):
bash src/download_real_data.sh          # needs your own ~/.kaggle credentials
python src/data_prep.py                 # auto-prefers real twcs.csv if present
python src/subsample_conversations.py --n 20000   # keep runtime reasonable

#    OR, for a fully offline demo instead:
# python src/make_synthetic_sample.py
# python src/data_prep.py

# 2. EDA + intent taxonomy sanity check
python notebooks/01_eda.py

# 3. Build the golden evaluation set (200 examples; --interactive hand-labels
#    the first N via CLI, see data/golden/README_golden_set.md for methodology)
python src/build_golden_set.py --interactive --hand-label-n 50

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

## Project structure
```
app.py                     optional Streamlit web UI (demo only, not required by the assignment)
src/
  config.py                 central config (paths, intent taxonomy, thresholds)
  make_synthetic_sample.py  generates a schema-identical synthetic dataset (offline demo path)
  download_real_data.sh     downloads the real Kaggle twcs.csv (needs your own login)
  data_prep.py              filters to AmazonHelp, pairs customer-brand tweets
  subsample_conversations.py subsamples the real dataset (assignment explicitly allows this)
  heuristics.py             cheap keyword bucket used only for EDA/sampling
  build_golden_set.py       stratified sampling plus hybrid hand-label/heuristic labelling
  baseline_classifiers.py   trivial plus TF-IDF/LogReg baselines
  llm_client.py             Groq API wrapper - offline mock fallback plus rate-limit backoff
  classify_intent.py        few-shot LLM intent classifier
  compare_baselines.py      3-way classifier comparison on held-out split
  retrieval.py              TF-IDF or embedding retriever for grounding
  draft_reply.py            retrieval-grounded reply drafting
  escalate.py               rule plus LLM escalation decision, always states a reason
  metrics.py                intent plus escalation automated metrics
  llm_judge.py              LLM-as-judge rubric for reply quality
  judge_agreement.py        judge vs hand-labelled human quality ratings
  run_eval.py               end-to-end evaluation harness over the golden set
  run_pipeline.py           interactive CLI: classify then draft then escalate
notebooks/01_eda.py         EDA justifying the intent taxonomy
data/raw/                   real twcs.csv, gitignored, or synthetic sample
data/processed/             cleaned conversation pairs
data/golden/                golden eval set plus hand-labelled judge-agreement set
reports/                    report.md, decision_log.md, failure_analysis.md
```

## Deliverables checklist
- [x] Runnable pipeline, reproducible in <15 min (`Quickstart` above)
- [x] Golden evaluation set - 200 examples, 50 hand-labelled + 150 heuristic (`data/golden/golden_eval_set.jsonl`, methodology in `data/golden/README_golden_set.md`)
- [x] Evaluation harness - automated metrics + LLM-as-judge + judge-human agreement (`src/run_eval.py`, `src/judge_agreement.py`)
- [x] Report - `reports/report.md` (problem framing, baselines, failure analysis, misleading-number section, next steps)
- [x] Decision log - `reports/decision_log.md` (13 decisions)

