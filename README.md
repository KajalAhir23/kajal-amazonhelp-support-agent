# Kajal — AI Support Agent for AmazonHelp

An AI agent that reads a customer's tweet to @AmazonHelp, classifies its intent,
drafts a reply grounded in how AmazonHelp has historically resolved similar issues,
and decides whether to auto-handle or escalate to a human — with a stated reason.

Built for the Hiver SDE Intern take-home assignment.

> Status: work in progress — this README is updated as the pipeline is built.
> Full reproduction instructions and headline results will be added in the final commit.

## Quickstart (placeholder — finalized later)
```bash
pip install -r requirements.txt
cp .env.example .env   # add your free Groq API key
python src/data_prep.py
python src/run_pipeline.py --brand AmazonHelp --n 50
```

## Project structure
```
src/            pipeline source code
data/raw/       raw + synthetic sample data (real twcs.csv is gitignored)
data/processed/ cleaned conversation pairs
data/golden/    hand-labelled golden evaluation set
reports/        report.md, decision_log.md, failure_analysis.md
notebooks/      exploration scripts
```
