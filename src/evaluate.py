# src/evaluate.py
# PURPOSE: Measure how good our retrieval and generation actually is.
# WHY THIS MATTERS: Without evaluation, you're guessing. With it, you can
# say "chunk_size 500 gave 20% better precision than 1000" — that's engineering.

import os
import json
import numpy as np
from retrieve import retrieve
from generate import generate_answer
from ingest import load_pdfs, chunk_documents
from embed import build_index

# ── GROUND TRUTH ─────────────────────────────────────────────────────────────
# These are hand-labeled query → expected source paper mappings.
# WHY hand-label? There's no shortcut for ground truth in retrieval eval.
# In production you'd use human annotators or a larger labeled dataset.
GROUND_TRUTH = [
    {
        "query": "What is retrieval augmented generation?",
        "relevant_sources": ["2005.11401v4.pdf", "2404.16130v2.pdf"]
    },
    {
        "query": "How does attention mechanism work?",
        "relevant_sources": ["1706.03762v7.pdf"]
    },
    {
        "query": "What is dense passage retrieval?",
        "relevant_sources": ["2004.04906v3.pdf"]
    },
    {
        "query": "How does FAISS perform similarity search?",
        "relevant_sources": ["1702.08734v1.pdf"]
    },
    {
        "query": "What is HyDE hypothetical document embedding?",
        "relevant_sources": ["2212.10496v1.pdf"]
    },
    {
        "query": "What is self-RAG?",
        "relevant_sources": ["2310.11511v1.pdf"]
    },
    {
        "query": "How do sentence transformers create embeddings?",
        "relevant_sources": ["1908.10084v1.pdf"]
    },
]
# ─────────────────────────────────────────────────────────────────────────────


def precision_at_k(retrieved_sources: list, relevant_sources: list, k: int) -> float:
    """
    Of the top-K retrieved chunks, what fraction came from relevant papers?
    Example: retrieved 5 chunks, 3 from correct paper → precision@5 = 0.6
    """
    retrieved_k = retrieved_sources[:k]
    hits = sum(1 for s in retrieved_k if s in relevant_sources)
    return hits / k


def recall_at_k(retrieved_sources: list, relevant_sources: list, k: int) -> float:
    """
    Of all relevant papers, what fraction did we retrieve at least one chunk from?
    Example: 2 relevant papers, retrieved chunks from 1 → recall@5 = 0.5
    """
    retrieved_k = set(retrieved_sources[:k])
    hits = sum(1 for s in relevant_sources if s in retrieved_k)
    return hits / len(relevant_sources)


def evaluate_retrieval(top_k: int = 5) -> dict:
    """Run all ground truth queries and compute average precision + recall."""
    precisions = []
    recalls    = []

    print(f"\n{'='*60}")
    print(f"RETRIEVAL EVALUATION  (top_k={top_k})")
    print(f"{'='*60}")

    for item in GROUND_TRUTH:
        query    = item["query"]
        relevant = item["relevant_sources"]

        results  = retrieve(query, top_k=top_k)
        retrieved_sources = [r["source"] for r in results]

        p = precision_at_k(retrieved_sources, relevant, top_k)
        r = recall_at_k(retrieved_sources, relevant, top_k)

        precisions.append(p)
        recalls.append(r)

        print(f"\nQuery   : {query}")
        print(f"Expected: {relevant}")
        print(f"Got     : {retrieved_sources}")
        print(f"P@{top_k}={p:.2f}  R@{top_k}={r:.2f}")

    avg_p = np.mean(precisions)
    avg_r = np.mean(recalls)

    print(f"\n{'='*60}")
    print(f"Average Precision@{top_k} : {avg_p:.4f}")
    print(f"Average Recall@{top_k}    : {avg_r:.4f}")
    print(f"{'='*60}")

    return {"precision": avg_p, "recall": avg_r, "top_k": top_k}


def evaluate_faithfulness(num_questions: int = 3) -> float:
    """
    Check if the generated answer only uses words/concepts from retrieved context.
    Simple heuristic: what % of answer words appear in the context?
    WHY this matters: hallucination = answer contains info NOT in context.
    """
    print(f"\n{'='*60}")
    print("FAITHFULNESS EVALUATION")
    print(f"{'='*60}")

    scores = []
    for item in GROUND_TRUTH[:num_questions]:
        result  = generate_answer(item["query"])
        answer  = result["answer"].lower().split()
        context = " ".join([c["text"] for c in result["chunks"]]).lower().split()
        context_set = set(context)

        # What fraction of answer words appear in context?
        overlap = sum(1 for w in answer if w in context_set)
        score   = overlap / len(answer) if answer else 0
        scores.append(score)

        print(f"\nQuery      : {item['query']}")
        print(f"Faithfulness: {score:.2f}  ({overlap}/{len(answer)} words in context)")

    avg = np.mean(scores)
    print(f"\nAverage Faithfulness: {avg:.4f}")
    return avg


def compare_chunk_sizes():
    """
    Compare precision@5 for chunk_size=500 vs chunk_size=1000.
    THIS IS YOUR INTERVIEW TALKING POINT:
    'I tested two chunking strategies and found X% difference in precision.'
    """
    print(f"\n{'='*60}")
    print("CHUNKING STRATEGY COMPARISON")
    print(f"{'='*60}")

    PAPERS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "papers")
    results    = {}

    for chunk_size in [500, 1000]:
        print(f"\n--- Testing chunk_size={chunk_size} ---")

        # Re-chunk with different size
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        docs = load_pdfs(PAPERS_DIR)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 10,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = []
        for doc in docs:
            for i, t in enumerate(splitter.split_text(doc["text"])):
                chunks.append({"text": t, "source": doc["source"], "chunk_id": i})

        print(f"Chunks created: {len(chunks)}")

        # Build temporary index
        index, metadata = build_index(chunks)

        # Evaluate precision using this index
        from sentence_transformers import SentenceTransformer
        import faiss, json
        precisions = []
        model = SentenceTransformer("all-MiniLM-L6-v2")

        for item in GROUND_TRUTH:
            qvec = model.encode([item["query"]])
            qvec = np.array(qvec, dtype=np.float32)
            dists, idxs = index.search(qvec, 5)
            sources = [metadata[i]["source"] for i in idxs[0] if i != -1]
            p = precision_at_k(sources, item["relevant_sources"], 5)
            precisions.append(p)

        avg_p = np.mean(precisions)
        results[chunk_size] = avg_p
        print(f"Average Precision@5: {avg_p:.4f}")

    print(f"\n{'='*60}")
    print(f"chunk_size=500  → Precision@5: {results[500]:.4f}")
    print(f"chunk_size=1000 → Precision@5: {results[1000]:.4f}")
    winner = 500 if results[500] >= results[1000] else 1000
    diff   = abs(results[500] - results[1000]) * 100
    print(f"\n✅ Winner: chunk_size={winner} by {diff:.1f}% better precision")
    print(f"{'='*60}")
    return results


if __name__ == "__main__":
    # 1. Retrieval quality
    evaluate_retrieval(top_k=5)

    # 2. Faithfulness
    evaluate_faithfulness(num_questions=3)

    # 3. Compare chunking strategies
    compare_chunk_sizes()