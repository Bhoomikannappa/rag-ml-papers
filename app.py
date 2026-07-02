# app.py
# PURPOSE: Streamlit chat UI for the RAG system
# Run with: streamlit run app.py

import streamlit as st
import sys
import os

# Add src/ to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from generate import generate_answer

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Q&A — ML Research Papers",
    page_icon="📚",
    layout="centered"
)

st.title("📚 RAG Q&A over ML Research Papers")
st.caption("Ask questions about RAG, Transformers, FAISS, DPR, and other ML papers. "
           "Answers are grounded in retrieved paper excerpts — not hallucinated.")

# ── SIDEBAR: Settings ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=5)
    st.markdown("---")
    st.markdown("### 📄 Papers indexed")
    papers_dir = os.path.join(os.path.dirname(__file__), "data", "papers")
    if os.path.exists(papers_dir):
        for f in os.listdir(papers_dir):
            if f.endswith(".pdf"):
                st.markdown(f"- {f}")
    st.markdown("---")
    st.markdown("**Tech stack:** FAISS · sentence-transformers · Groq (LLaMA 3.1) · Streamlit")

# ── CHAT HISTORY ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📄 View sources"):
                for s in msg["sources"]:
                    st.markdown(f"- `{s}`")

# ── CHAT INPUT ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask something about RAG, attention, FAISS, DPR..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant chunks and generating answer..."):
            result = generate_answer(prompt, top_k=top_k)
            st.markdown(result["answer"])

            with st.expander("📄 View sources"):
                for s in result["sources"]:
                    st.markdown(f"- `{s}`")

            with st.expander("🔍 View retrieved chunks (debug)"):
                for i, c in enumerate(result["chunks"]):
                    st.markdown(f"**Chunk {i+1}** — `{c['source']}` (distance: {c['distance']:.4f})")
                    st.text(c["text"][:300] + "...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })