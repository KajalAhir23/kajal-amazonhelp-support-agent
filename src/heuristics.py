"""Cheap keyword heuristic used for EDA and for stratifying the golden-set
sample. This is NOT the classifier (see src/classify_intent.py) — it exists
only so sampling isn't purely random and rare intents get enough coverage.
"""
from config import INTENT_LABELS

KEYWORD_MAP = {
    "delivery_delay_or_lost": ["delay", "late", "hasn't arrived", "tracking", "lost", "where is"],
    "refund_or_return_request": ["refund", "return", "money back"],
    "order_or_item_issue": ["wrong item", "wrong order", "missing item", "incorrect"],
    "account_or_login_issue": ["log in", "login", "password", "account", "sign in"],
    "billing_or_charge_dispute": ["charged twice", "charge", "billed", "overcharged"],
    "product_defect_or_damage": ["broken", "damaged", "defective", "cracked", "doesn't work"],
    "cancellation_request": ["cancel"],
    "general_inquiry_or_other": [],  # fallback bucket
}


def keyword_bucket(text: str) -> str:
    text_l = text.lower()
    for intent, kws in KEYWORD_MAP.items():
        if any(kw in text_l for kw in kws):
            return intent
    return "general_inquiry_or_other"
