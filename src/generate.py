# src/generate.py
# PURPOSE: Take a user query → retrieve relevant chunks → generate answer via Groq
# This is the full RAG pipeline in one file.

import os
from groq import Groq
from dotenv import load_dotenv
from retrieve import retrieve

# Load GROQ_API_KEY from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.1-8b-instant" 
# WHY Groq + LLaMA3? Free, extremely fast (500+ tokens/sec).
# Same quality as GPT-3.5 for Q&A tasks. Perfect for RAG demos.


def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Combine retrieved chunks + user question into one prompt.
    WHY this structure? LLMs follow instructions better when context
    comes BEFORE the question, and when told explicitly to use only
    the provided context (reduces hallucination).
    """
    context = "\n\n---\n\n".join([
        f"Source: {c['source']}\n{c['text']}"
        for c in chunks
    ])

    prompt = f"""You are an expert ML research assistant. 
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information in the provided papers."
Do NOT make up information.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    return prompt


def generate_answer(query: str, top_k: int = 5) -> dict:
    """Full RAG pipeline: retrieve → augment → generate."""

    # Step 1: Retrieve relevant chunks
    chunks = retrieve(query, top_k=top_k)

    # Step 2: Build augmented prompt
    prompt = build_prompt(query, chunks)

    # Step 3: Generate answer with Groq
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        # WHY low temperature (0.2)? We want factual, consistent answers
        # not creative ones. Higher temp = more random = more hallucination.
        max_tokens=512
    )

    answer = response.choices[0].message.content

    return {
        "query"   : query,
        "answer"  : answer,
        "sources" : list(set([c["source"] for c in chunks])),
        "chunks"  : chunks
    }


if __name__ == "__main__":
    test_questions = [
        "What is retrieval augmented generation and why is it useful?",
        "How does the attention mechanism work in transformers?",
        "What is Dense Passage Retrieval?"
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print(f"{'='*60}")
        result = generate_answer(question)
        print(f"\nA: {result['answer']}")
        print(f"\n📄 Sources: {', '.join(result['sources'])}")