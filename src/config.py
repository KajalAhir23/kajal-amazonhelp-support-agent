"""
Kajal — AI Support Agent for AmazonHelp
Central configuration for the pipeline.
"""
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = "Kajal"
BRAND = "AmazonHelp"

# --- Paths ---
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
DATA_GOLDEN_DIR = "data/golden"
REPORTS_DIR = "reports"

RAW_TWCS_PATH = os.path.join(DATA_RAW_DIR, "twcs.csv")
SYNTHETIC_SAMPLE_PATH = os.path.join(DATA_RAW_DIR, "twcs_sample.csv")
CONVERSATIONS_PATH = os.path.join(DATA_PROCESSED_DIR, "conversations.jsonl")
GOLDEN_SET_PATH = os.path.join(DATA_GOLDEN_DIR, "golden_eval_set.jsonl")

# --- LLM config (Groq — free tier) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")

# --- Embedding config (local, free) ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Intent taxonomy (defined from data in notebooks/01_eda.py) ---
INTENT_LABELS = [
    "delivery_delay_or_lost",
    "refund_or_return_request",
    "order_or_item_issue",
    "account_or_login_issue",
    "billing_or_charge_dispute",
    "product_defect_or_damage",
    "cancellation_request",
    "general_inquiry_or_other",
]

# --- Escalation thresholds ---
LOW_CONFIDENCE_THRESHOLD = 0.55
ESCALATION_KEYWORDS = [
    "lawyer", "sue", "scam", "fraud", "unacceptable", "furious",
    "never again", "worst experience", "manager", "compensation",
]
