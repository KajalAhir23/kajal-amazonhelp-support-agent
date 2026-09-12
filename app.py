"""
Kajal — Streamlit UI

A simple browser-based demo on top of the existing pipeline (classify ->
retrieve -> draft -> escalate). Run from the repo root:

    streamlit run app.py

This is NOT required by the assignment (which only asks for a runnable
CLI pipeline) - it's an optional visual layer for demoing the agent.
"""
import sys
import json
import random

import streamlit as st

sys.path.insert(0, "src")
from config import PROJECT_NAME, BRAND, GOLDEN_SET_PATH
from classify_intent import classify
from retrieval import build_retriever
from draft_reply import draft
from escalate import decide
from llm_client import has_real_llm

st.set_page_config(page_title=f"{PROJECT_NAME} - AI Support Agent", page_icon="🛠️", layout="centered")


@st.cache_resource
def get_retriever():
    return build_retriever()


@st.cache_data
def load_sample_messages(n=8):
    try:
        golden = [json.loads(l) for l in open(GOLDEN_SET_PATH, encoding="utf-8")]
        random.seed(0)
        return [g["customer_text"] for g in random.sample(golden, min(n, len(golden)))]
    except FileNotFoundError:
        return []


def analyze(message: str) -> dict:
    retriever = get_retriever()
    intent = classify(message)
    draft_result = draft(message, retriever=retriever)
    retrieval_conf = draft_result["grounding_scores"][0] if draft_result["grounding_scores"] else 0.0
    escalation = decide(message, intent, draft_result["drafted_reply"], retrieval_conf)
    return {
        "intent": intent,
        "reply": draft_result["drafted_reply"],
        "grounded_in": draft_result["grounding_examples"],
        "grounding_score": retrieval_conf,
        "escalate": escalation["escalate"],
        "reason": escalation["reason"],
    }


st.title(f"🛠️ {PROJECT_NAME}")
st.caption(f"AI Support Agent for {BRAND} - classify -> ground a reply in history -> decide auto-handle vs. escalate")

if not has_real_llm():
    st.warning(
        "No GROQ_API_KEY detected - running in offline fallback mode "
        "(keyword-based classification, closest historical reply, rule-only escalation). "
        "Add a key to `.env` for real LLM-powered results.",
        icon="⚠️",
    )

samples = load_sample_messages()
if samples:
    st.write("**Try a real example from the golden set:**")
    cols = st.columns(2)
    for i, s in enumerate(samples):
        if cols[i % 2].button(s[:60] + ("..." if len(s) > 60 else ""), key=f"sample_{i}", use_container_width=True):
            st.session_state["message_input"] = s

message = st.text_area(
    "Customer message",
    key="message_input",
    height=100,
    placeholder="e.g. My order never arrived and tracking hasn't updated in 5 days...",
)

if st.button("Analyze", type="primary", use_container_width=True):
    if not message.strip():
        st.error("Please enter a customer message first.")
    else:
        with st.spinner("Running classify -> retrieve -> draft -> escalate..."):
            result = analyze(message.strip())

        st.divider()

        st.subheader("Predicted intent")
        st.code(result["intent"], language=None)

        st.subheader("Drafted reply")
        st.info(result["reply"])

        st.subheader("Grounded in (historical conversations)")
        st.write(f"Conversation IDs: `{', '.join(result['grounded_in'])}`  "
                 f"(top similarity score: {result['grounding_score']:.3f})")

        st.subheader("Decision")
        if result["escalate"]:
            st.error(f"🚨 ESCALATE - {result['reason']}")
        else:
            st.success(f"✅ AUTO-HANDLE - {result['reason']}")

st.divider()
st.caption(
    "Built for the Hiver SDE Intern take-home assignment. "
    "This UI is an optional extra - see `reports/report.md` for the full evaluation and methodology."
)
