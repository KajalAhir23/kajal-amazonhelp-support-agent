"""
Loads twcs.csv (real if present, else the synthetic sample), filters to
BRAND (AmazonHelp), and reconstructs customer -> brand reply pairs using
the in_response_to_tweet_id / response_tweet_id threading columns.

Output: data/processed/conversations.jsonl, one JSON object per pair:
{
  "conv_id": str,
  "customer_text": str,
  "brand_reply": str,
  "customer_tweet_id": str,
  "brand_tweet_id": str
}
"""
import json
import os
import re
import pandas as pd

from config import RAW_TWCS_PATH, SYNTHETIC_SAMPLE_PATH, CONVERSATIONS_PATH, BRAND, DATA_PROCESSED_DIR


def load_raw():
    if os.path.exists(RAW_TWCS_PATH):
        print(f"Using REAL dataset: {RAW_TWCS_PATH}")
        path = RAW_TWCS_PATH
    elif os.path.exists(SYNTHETIC_SAMPLE_PATH):
        print(f"Real twcs.csv not found — using synthetic sample: {SYNTHETIC_SAMPLE_PATH}")
        print("Run src/download_real_data.sh with your own Kaggle credentials for the real thing.")
        path = SYNTHETIC_SAMPLE_PATH
    else:
        raise FileNotFoundError(
            "No dataset found. Run `python src/make_synthetic_sample.py` "
            "or `bash src/download_real_data.sh`."
        )
    df = pd.read_csv(path, dtype=str)
    df["inbound"] = df["inbound"].astype(str).str.lower() == "true"
    return df


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"https?://\S+", "", text)          # strip links
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_conversation_pairs(df: pd.DataFrame, brand: str = BRAND) -> list:
    """Pair each customer (inbound) tweet with the brand's direct reply."""
    by_id = df.set_index("tweet_id", drop=False)
    brand_replies = df[(~df["inbound"]) & (df["author_id"] == brand)]

    pairs = []
    for _, reply_row in brand_replies.iterrows():
        parent_id = reply_row.get("in_response_to_tweet_id")
        if pd.isna(parent_id) or parent_id == "" or parent_id not in by_id.index:
            continue
        parent = by_id.loc[parent_id]
        if isinstance(parent, pd.DataFrame):  # duplicate ids guard
            parent = parent.iloc[0]
        if not parent["inbound"]:
            continue  # only keep genuine customer -> brand pairs

        pairs.append({
            "conv_id": f"{parent['tweet_id']}_{reply_row['tweet_id']}",
            "customer_text": clean_text(parent["text"]),
            "brand_reply": clean_text(reply_row["text"]),
            "customer_tweet_id": str(parent["tweet_id"]),
            "brand_tweet_id": str(reply_row["tweet_id"]),
        })
    return pairs


def main():
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    df = load_raw()
    pairs = build_conversation_pairs(df, brand=BRAND)

    with open(CONVERSATIONS_PATH, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Built {len(pairs)} {BRAND} conversation pairs -> {CONVERSATIONS_PATH}")


if __name__ == "__main__":
    main()
