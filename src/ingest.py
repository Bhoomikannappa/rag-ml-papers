# src/ingest.py
# PURPOSE: Load all PDFs and split them into chunks for embedding
# WHY CHUNKING? LLMs have a token limit — we can't feed a whole paper.
# We split into small overlapping pieces so no context is lost at boundaries.

import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── CONFIG ───────────────────────────────────────────────────────────────────
PAPERS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "papers")
CHUNK_SIZE  = 500   # number of characters per chunk
CHUNK_OVERLAP = 50  # overlap between chunks so context isn't cut off
# ─────────────────────────────────────────────────────────────────────────────


def load_pdfs(papers_dir: str) -> list[dict]:
    """Read every PDF and return a list of {text, source} dicts."""
    documents = []
    pdf_files = [f for f in os.listdir(papers_dir) if f.endswith(".pdf")]

    if not pdf_files:
        print("⚠️  No PDFs found in", papers_dir)
        return documents

    for filename in pdf_files:
        filepath = os.path.join(papers_dir, filename)
        try:
            reader = PdfReader(filepath)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() or ""
            documents.append({
                "text": full_text,
                "source": filename
            })
            print(f"✅ Loaded: {filename}  ({len(reader.pages)} pages)")
        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split each document's text into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
        # WHY THIS ORDER? Try paragraph breaks first, then sentences,
        # then words — so we never cut mid-sentence if avoidable.
    )

    all_chunks = []
    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": doc["source"],
                "chunk_id": i
            })

    return all_chunks


if __name__ == "__main__":
    print("\n=== Loading PDFs ===")
    docs = load_pdfs(PAPERS_DIR)
    print(f"\nTotal documents loaded: {len(docs)}")

    print("\n=== Chunking ===")
    chunks = chunk_documents(docs)
    print(f"Total chunks created: {len(chunks)}")

    print("\n=== Sample Chunk ===")
    if chunks:
        print(f"Source : {chunks[0]['source']}")
        print(f"Chunk 0: {chunks[0]['text'][:300]}...")