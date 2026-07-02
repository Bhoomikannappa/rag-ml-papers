# src/retrieve.py
# PURPOSE: Given a user query, find the most relevant chunks from FAISS
# HOW? Embed the query using same model → find nearest vectors → return text

import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── CONFIG ───────────────────────────────────────────────────────────────────
INDEX_DIR  = os.path.join(os.path.dirname(__file__), "..", "faiss_index")
MODEL_NAME = "all-MiniLM-L6-v2"  # MUST be same model used in embed.py
TOP_K      = 5  # how many chunks to retrieve per query
# ─────────────────────────────────────────────────────────────────────────────

# Load once at import time so we don't reload on every query
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))

with open(os.path.join(INDEX_DIR, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed the query and return top_k most similar chunks.
    WHY same model? The query vector and chunk vectors must live in the
    same vector space — only possible if encoded with the same model.
    """
    query_vector = model.encode([query], convert_to_numpy=True)
    query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)

    # FAISS returns distances + indices of nearest neighbors
    distances, indices = index.search(query_vector, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue  # FAISS returns -1 if not enough results
        results.append({
            "text"    : metadata[idx]["text"],
            "source"  : metadata[idx]["source"],
            "chunk_id": metadata[idx]["chunk_id"],
            "distance": float(dist)
            # LOWER distance = MORE similar (L2 distance)
        })
    return results


if __name__ == "__main__":
    test_queries = [
        "What is retrieval augmented generation?",
        "How does attention mechanism work in transformers?",
        "What is the difference between dense and sparse retrieval?"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")
        results = retrieve(query)
        for i, r in enumerate(results):
            print(f"\n[{i+1}] Source: {r['source']} | Distance: {r['distance']:.4f}")
            print(f"     {r['text'][:200]}...")