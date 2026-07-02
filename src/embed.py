# src/embed.py
# PURPOSE: Convert all text chunks into vectors and store in FAISS
# WHY EMBEDDINGS? Computers can't compare text directly.
# We convert text → numbers (vectors) so similar meaning = similar numbers.
# Then FAISS lets us search 1411 vectors in milliseconds.

import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ingest import load_pdfs, chunk_documents

# ── CONFIG ───────────────────────────────────────────────────────────────────
PAPERS_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "papers")
INDEX_DIR   = os.path.join(os.path.dirname(__file__), "..", "faiss_index")
MODEL_NAME  = "all-MiniLM-L6-v2"
# WHY this model? Fast, free, runs locally, 384-dim vectors.
# Good enough for semantic search. Used by thousands of RAG systems in prod.
# ─────────────────────────────────────────────────────────────────────────────


def build_index(chunks: list[dict]):
    """Embed all chunks and save FAISS index + metadata to disk."""

    print(f"\n🔄 Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Extract just the text from each chunk for embedding
    texts = [chunk["text"] for chunk in chunks]

    print(f"🔄 Embedding {len(texts)} chunks... (this takes 1-2 minutes)")
    embeddings = model.encode(texts, show_progress_bar=True)
    # embeddings shape: (1411, 384) — each chunk becomes 384 numbers

    # ── Build FAISS index ──────────────────────────────────────────────────
    dimension = embeddings.shape[1]  # 384
    index = faiss.IndexFlatL2(dimension)
    # WHY IndexFlatL2? L2 = Euclidean distance. Simple, exact, perfect for
    # our size. For millions of vectors you'd use IndexIVFFlat (approximate).

    index.add(np.array(embeddings, dtype=np.float32))
    print(f"✅ FAISS index built with {index.ntotal} vectors")

    # ── Save to disk ───────────────────────────────────────────────────────
    os.makedirs(INDEX_DIR, exist_ok=True)

    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))

    # Save chunk metadata separately so we can retrieve original text later
    metadata = [{"text": c["text"], "source": c["source"], "chunk_id": c["chunk_id"]}
                for c in chunks]
    with open(os.path.join(INDEX_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved index  → faiss_index/index.faiss")
    print(f"✅ Saved metadata → faiss_index/metadata.json")
    return index, metadata


if __name__ == "__main__":
    print("=== Step 1: Load & Chunk PDFs ===")
    docs   = load_pdfs(PAPERS_DIR)
    chunks = chunk_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    print("\n=== Step 2: Embed & Index ===")
    index, metadata = build_index(chunks)

    print(f"\n🎉 Done! {index.ntotal} vectors stored in faiss_index/")
    print("You can now run retrieve.py to search them.")