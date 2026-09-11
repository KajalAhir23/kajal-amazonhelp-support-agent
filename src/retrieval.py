"""
Retrieves the most similar historically-resolved AmazonHelp conversations
for a new customer message, so the reply drafter can ground its answer in
"how this brand has actually handled similar issues" rather than
hallucinating a generic response.

Two backends:
  - TfidfRetriever        : default here. Pure sklearn, no model download,
                            works fully offline — used because this sandbox
                            cannot reach huggingface.co to pull model weights.
  - EmbeddingRetriever    : sentence-transformers based (all-MiniLM-L6-v2),
                            better semantic matching. Drop-in replacement —
                            just set RETRIEVAL_BACKEND=embedding in .env on
                            a machine with normal internet access.
"""
import json
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "src")
from config import CONVERSATIONS_PATH, EMBEDDING_MODEL

RETRIEVAL_BACKEND = os.getenv("RETRIEVAL_BACKEND", "tfidf")


class TfidfRetriever:
    def __init__(self, corpus: list):
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
        self.matrix = self.vectorizer.fit_transform([c["customer_text"] for c in corpus])

    def top_k(self, query: str, k: int = 3):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        idx = np.argsort(-sims)[:k]
        return [(self.corpus[i], float(sims[i])) for i in idx]


class EmbeddingRetriever:
    """Semantic retriever using sentence-transformers. Requires internet
    access to huggingface.co on first run to download model weights."""
    def __init__(self, corpus: list):
        from sentence_transformers import SentenceTransformer
        self.corpus = corpus
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.embeddings = self.model.encode(
            [c["customer_text"] for c in corpus], normalize_embeddings=True
        )

    def top_k(self, query: str, k: int = 3):
        q_emb = self.model.encode([query], normalize_embeddings=True)
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        idx = np.argsort(-sims)[:k]
        return [(self.corpus[i], float(sims[i])) for i in idx]


def build_retriever(backend: str = None):
    backend = backend or RETRIEVAL_BACKEND
    corpus = [json.loads(l) for l in open(CONVERSATIONS_PATH, encoding="utf-8")]
    if backend == "embedding":
        return EmbeddingRetriever(corpus)
    return TfidfRetriever(corpus)


if __name__ == "__main__":
    retriever = build_retriever()
    query = "my package never arrived and tracking says nothing"
    print(f"Query: {query}\n")
    for conv, score in retriever.top_k(query, k=3):
        print(f"  score={score:.3f}  customer=\"{conv['customer_text'][:60]}...\"")
        print(f"           historical_reply=\"{conv['brand_reply'][:70]}...\"\n")
