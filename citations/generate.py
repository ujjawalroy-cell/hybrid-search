from __future__ import annotations

import os
import requests
from dotenv import load_dotenv
from typing import Any, List, Union

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_URL", "http://localhost:11434").strip().strip('"').strip("'").rstrip("/")
DEFAULT_MODEL = "llama3.2"


def format_context(context_data: Any) -> str:
    """
    Normalizes various retrieved result formats (list of dicts, search outputs,
    list of strings, or raw string) into a cohesive context string for the prompt.
    """
    if isinstance(context_data, str):
        return context_data.strip()

    passages: List[str] = []

    if isinstance(context_data, dict):
        # Case: output from chunking.search() with keys 'dense' and 'sparse'
        for key in ("dense", "sparse"):
            items = context_data.get(key, [])
            for item in items:
                if isinstance(item, dict) and "text" in item:
                    passages.append(item["text"])
                elif isinstance(item, str):
                    passages.append(item)

    elif isinstance(context_data, list):
        for item in context_data:
            if isinstance(item, dict):
                # Format from hybrid_retrieve(): {"id": ..., "text": ..., "rerank_score": ...}
                text = item.get("text", "")
                heading = item.get("metadata", {}).get("section_heading", "") if isinstance(item.get("metadata"), dict) else ""
                cid = item.get("id", "")
                header = f"[{cid}]" + (f" ({heading})" if heading else "")
                if text:
                    passages.append(f"{header}:\n{text}")
            elif hasattr(item, "page_content"):
                # Langchain Document
                passages.append(item.page_content)
            elif isinstance(item, str):
                passages.append(item)

    if not passages:
        return "No relevant context found."

    return "\n\n---\n\n".join(passages)


def generate_answer(
    query: str,
    retrieved_results: Any,
    model: str = DEFAULT_MODEL,
    system_instruction: str | None = None,
    timeout: int = 120,
) -> str:
    """
    Generates a grounded answer to the user query given the retrieved context passages
    using the local Ollama LLM.
    """
    context_text = format_context(retrieved_results)

    default_system = (
        "You are an accurate, helpful AI assistant. Answer the question based on the provided "
        "context. If the context does not contain enough information to answer the question, "
        "state that clearly. Keep the answer well-structured and concise."
    )
    system = system_instruction or default_system

    prompt = f"""[Context Information]
{context_text}

[Question]
{query}

[Instructions]
Provide a detailed and direct answer based on the context above. Cite relevant sections or documents if applicable.

[Answer]
"""

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": model,
                "system": system,
                "prompt": prompt,
                "stream": False,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to generate answer from Ollama at {OLLAMA_HOST} (model: {model}): {e}"
        print(f"Error in generate_answer: {error_msg}")
        return f"Error generating answer: {e}"


RETRIEVAL_CONFIDENCE_THRESHOLD = 0.35


def rag_answer(
    query: str,
    top_k: int = 5,
    use_sparse: bool = True,
    retrieval_threshold: float = RETRIEVAL_CONFIDENCE_THRESHOLD,
    model: str = DEFAULT_MODEL,
    filter_document: Optional[str] = None,
    strategy: Optional[str] = None,
) -> dict:
    """
    End-to-end RAG execution:
    1. Retrieves relevant passages (Hybrid dense + sparse or dense-only)
    2. Computes retrieval confidence
    3. Guards against hallucination if confidence < retrieval_threshold
    4. Generates cited answer via local Ollama LLM
    """
    import math
    from retrievals.retrieval import hybrid_retrieve, dense_search, hydrate

    if use_sparse:
        chunks = hybrid_retrieve(
            query,
            final_top_n=top_k,
            filter_document=filter_document,
            strategy=strategy,
        )
    else:
        dense_raw = dense_search(
            query,
            k=top_k,
            filter_document=filter_document,
            strategy=strategy,
        )
        content = hydrate([d["id"] for d in dense_raw])
        chunks = [
            {"id": d["id"], "score": d["score"], **content.get(d["id"], {})}
            for d in dense_raw if d["id"] in content
        ]

    if not chunks:
        return {
            "query": query,
            "status": "insufficient_context",
            "answer": "No relevant context found in the knowledge base.",
            "chunks_used": [],
            "retrieval_confidence": 0.0,
        }

    top_score = chunks[0].get("rerank_score", chunks[0].get("score", 0.0))
    # Normalize cross-encoder logit using sigmoid, or clamp similarity score
    if "rerank_score" in chunks[0]:
        confidence = round(1.0 / (1.0 + math.exp(-top_score)), 3)
    else:
        confidence = round(max(0.0, min(1.0, float(top_score))), 3)

    if confidence < retrieval_threshold:
        return {
            "query": query,
            "status": "insufficient_context",
            "answer": f"Retrieval confidence ({confidence:.2f}) is below threshold ({retrieval_threshold:.2f}). Refusing to answer to avoid hallucination.",
            "chunks_used": chunks,
            "retrieval_confidence": confidence,
        }

    answer = generate_answer(query, chunks, model=model)
    return {
        "query": query,
        "status": "answered",
        "answer": answer,
        "chunks_used": chunks,
        "retrieval_confidence": confidence,
    }


if __name__ == "__main__":
    sample_context = [
        {"id": "test::1", "text": "Hybrid search combines dense vector similarity search with sparse lexical BM25 search for optimal retrieval quality.", "metadata": {"section_heading": "Overview"}}
    ]
    ans = generate_answer("What does hybrid search combine?", sample_context)
    print("Generated Answer:\n", ans)
