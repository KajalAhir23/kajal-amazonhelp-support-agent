"""
Generates a small synthetic dataset in the EXACT schema of Kaggle's
'Customer Support on Twitter' (twcs.csv), scoped to @AmazonHelp-style
conversations, so the pipeline can be built and tested without needing
a Kaggle login inside a sandboxed environment.

Schema (matches twcs.csv):
tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id

Swap this out for the real file at data/raw/twcs.csv (see README) —
no other code changes needed; src/data_prep.py reads the same columns.
"""
import csv
import random
from datetime import datetime, timedelta

random.seed(42)

BRAND = "AmazonHelp"

# (customer_msg, brand_reply, intent) — used to generate realistic-looking
# variations. Intent label is only used to shape content here; the real
# labelling pipeline in later commits does NOT read this field.
TEMPLATES = [
    ("Hi @AmazonHelp my package #{oid} was supposed to arrive 3 days ago and "
     "tracking hasn't updated at all. Where is it??",
     "Hi there, we're sorry for the delay! Please DM us your order number "
     "and zip code so we can look into the tracking for you. ^KL",
     "delivery_delay_or_lost"),

    ("@AmazonHelp order {oid} arrived but it's the wrong item entirely. "
     "I ordered a blender and got headphones. Need this fixed asap",
     "So sorry about that mix-up! Please DM your order number and we'll "
     "arrange a free return and get the correct item shipped out. ^SM",
     "order_or_item_issue"),

    ("@AmazonHelp I was charged twice for order {oid}, can someone explain "
     "why there are two charges on my card for the same item?",
     "We apologize for the confusion — please DM your order ID and the "
     "last 4 digits of the card used so we can review the charges. ^RT",
     "billing_or_charge_dispute"),

    ("Can't log into my account, it keeps saying my password is wrong even "
     "after I reset it 3 times @AmazonHelp this is so frustrating",
     "Sorry for the trouble logging in! Please try resetting from a "
     "private/incognito window, and DM us if it still doesn't work. ^JD",
     "account_or_login_issue"),

    ("@AmazonHelp the product I received for order {oid} is completely "
     "broken, screen is cracked out of the box.",
     "That's not the experience we want for you — please DM your order "
     "number and a photo of the damage so we can send a replacement. ^KL",
     "product_defect_or_damage"),

    ("I want to cancel order {oid}, I placed it by mistake, how do I do "
     "this @AmazonHelp before it ships?",
     "No problem, you can cancel from Your Orders if it hasn't shipped "
     "yet. If the option isn't there, DM us the order number. ^SM",
     "cancellation_request"),

    ("@AmazonHelp requesting a refund for order {oid}, item doesn't match "
     "the description at all, very disappointed.",
     "We're sorry to hear that! Please DM your order number and we'll "
     "start a refund review right away. ^RT",
     "refund_or_return_request"),

    ("Does @AmazonHelp support gift wrapping for electronics orders? "
     "Just curious before I place order {oid}",
     "Great question! Gift wrap availability shows on the product page "
     "at checkout for eligible items. Let us know if you need more help! ^JD",
     "general_inquiry_or_other"),

    ("@AmazonHelp this is the THIRD time my order {oid} has been delayed. "
     "Absolutely unacceptable, I want a manager to call me.",
     "We completely understand your frustration and apologize for the "
     "repeated delays. Please DM your order number so we can escalate "
     "this internally. ^KL",
     "delivery_delay_or_lost"),

    ("Refund for order {oid} still hasn't shown up after 2 weeks "
     "@AmazonHelp, this feels like a scam at this point.",
     "We're sorry for the wait — refunds can take 3-5 business days once "
     "processed. Please DM your order number so we can check the status. ^SM",
     "refund_or_return_request"),
]

CUSTOMER_HANDLES = [f"cust_{i}" for i in range(1, 401)]


def build_rows(n_conversations=220):
    rows = []
    tweet_id = 1
    base_time = datetime(2017, 10, 1)

    for i in range(n_conversations):
        cust_msg, brand_reply, _intent = random.choice(TEMPLATES)
        oid = f"{random.randint(100000, 999999)}"
        cust_msg = cust_msg.format(oid=oid)
        customer = random.choice(CUSTOMER_HANDLES)
        t0 = base_time + timedelta(minutes=random.randint(0, 500000))
        t1 = t0 + timedelta(minutes=random.randint(5, 240))

        cust_tid = tweet_id
        brand_tid = tweet_id + 1
        tweet_id += 2

        rows.append({
            "tweet_id": cust_tid,
            "author_id": customer,
            "inbound": "True",
            "created_at": t0.strftime("%a %b %d %H:%M:%S +0000 %Y"),
            "text": cust_msg,
            "response_tweet_id": str(brand_tid),
            "in_response_to_tweet_id": "",
        })
        rows.append({
            "tweet_id": brand_tid,
            "author_id": BRAND,
            "inbound": "False",
            "created_at": t1.strftime("%a %b %d %H:%M:%S +0000 %Y"),
            "text": brand_reply,
            "response_tweet_id": "",
            "in_response_to_tweet_id": str(cust_tid),
        })
    return rows


def main(out_path="data/raw/twcs_sample.csv"):
    rows = build_rows()
    fieldnames = ["tweet_id", "author_id", "inbound", "created_at", "text",
                  "response_tweet_id", "in_response_to_tweet_id"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows ({len(rows)//2} conversations) to {out_path}")


if __name__ == "__main__":
    main()
