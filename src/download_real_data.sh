#!/bin/bash
# Downloads the REAL Kaggle 'Customer Support on Twitter' dataset.
# Requires your own Kaggle account + API token (~/.kaggle/kaggle.json).
# See: https://www.kaggle.com/docs/api#authentication
#
# This is a separate step because it needs YOUR Kaggle login — it cannot
# be fetched from a sandboxed build environment.

set -e
pip install -q kaggle
mkdir -p data/raw
kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw --unzip
echo "Done. Real data at data/raw/twcs.csv"
echo "Pipeline will automatically prefer twcs.csv over the synthetic sample if present."
