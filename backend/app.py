"""
FastAPI Service for the Hybrid RAG Pipeline.
============================================

Endpoints:
    POST /v1/ask        Submit a question -> receives grounded answer + source citations + confidence
    GET  /v1/documents  List all indexed documents & chunk counts in ChromaDB
    GET  /v1/strategies List available chunking strategies
    POST /v1/ingest     Ingest new chunks into ChromaDB (dense) + BM25 (sparse)
    GET  /health        Health check & total chunks count
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional, List

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

import chunking.chunking as chunking_mod
import citations.generate as generation_mod
import ingestion.Ingest as ingest_mod
import splitter.Splitter as splitter_mod


app = FastAPI()

# CORS middleware for frontend integrations (Streamlit / React / etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================================
# Pydantic Request & Response Schemas
# =========================================================================

class AskRequest(BaseModel):
    question: str = Field(..., description="The user's question.", min_length=1)
    filter_document: Optional[str] = Field(None, description="Optional document name to restrict retrieval context to.")
    strategy: Optional[str] = Field(None, description="Optional chunking strategy to target.")
    use_sparse: bool = Field(True, description="True for hybrid retrieval (dense + BM25 + RRF), False for dense-only.")
    top_k: int = Field(5, description="Number of top reranked chunks to feed as context into the LLM.", ge=1, le=20)
    retrieval_threshold: float = Field(
        generation_mod.RETRIEVAL_CONFIDENCE_THRESHOLD,
        description="Minimum retrieval confidence score required before answering to prevent hallucination.",
        ge=0.0,
        le=1.0,
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What is the relationship between humans and the environment?",
                "filter_document": "Energy Resources.pdf",
                "use_sparse": True,
                "top_k": 4,
                "retrieval_threshold": 0.35,
            }
        }
    )


class SourceChunk(BaseModel):
    n: int = Field(..., description="Citation index (1-based).")
    id: str = Field(..., description="Unique chunk identifier.")
    source_document: str = Field(..., description="Origin document filename.")
    section_heading: str = Field("", description="Section or chapter heading.")
    score: Optional[float] = Field(None, description="Cross-encoder or similarity score.")
    text: Optional[str] = Field(None, description="Snippet of chunk text.")


class ConfidenceBreakdown(BaseModel):
    retrieval_confidence: float = Field(..., description="Relevance confidence score (0-1).")
    composite: float = Field(..., description="Overall confidence score.")


class AskResponse(BaseModel):
    query: str
    status: str = Field(..., description="'answered' or 'insufficient_context'")
    answer: Optional[str] = Field(None, description="Generated answer from the LLM.")
    sources: List[SourceChunk] = Field(default_factory=list, description="Retrieved source chunks used as context.")
    confidence: ConfidenceBreakdown


class DocumentInfo(BaseModel):
    source_document: str
    chunking_strategy: str
    chunk_count: int


class IngestChunk(BaseModel):
    text: str = Field(..., min_length=1)
    source_document: str = Field(..., min_length=1)
    chunk_index: int = Field(0, ge=0)
    section_heading: str = Field("")
    chunking_strategy: str = Field("default")


class IngestRequest(BaseModel):
    chunks: List[IngestChunk] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    status: str
    chunks_submitted: int
    chunks_indexed: int
    duplicates_skipped: int


class UploadDocumentResponse(BaseModel):
    status: str
    filename: str
    pages_loaded: int
    chunks_created: int
    chunks_indexed: int
    duplicates_skipped: int
    total_chunks_indexed: int


# =========================================================================
# API Endpoints
# =========================================================================

@app.post("/v1/ask", response_model=AskResponse, tags=["RAG"])
def ask(req: AskRequest) -> AskResponse:
    """
    Core RAG Pipeline:
    1. Hybrid Retrieval: Queries Chroma (dense) + BM25 (sparse), fuses via RRF, and reranks with Cross-Encoder.
    2. Guardrail Check: Compares retrieval confidence against retrieval_threshold to avoid hallucination.
    3. Grounded Generation: Synthesizes a factual answer using the local Ollama LLM.
    """
    # Execute end-to-end RAG
    result = generation_mod.rag_answer(
        query=req.question,
        top_k=req.top_k,
        use_sparse=req.use_sparse,
        retrieval_threshold=req.retrieval_threshold,
        filter_document=req.filter_document,
        strategy=req.strategy,
    )

    # Format source passages
    sources = []
    for i, c in enumerate(result.get("chunks_used", []), start=1):
        meta = c.get("metadata", {}) if isinstance(c.get("metadata"), dict) else {}
        sources.append(
            SourceChunk(
                n=i,
                id=c.get("id", f"chunk_{i}"),
                source_document=meta.get("source_document", c.get("source_document", "unknown")),
                section_heading=meta.get("section_heading", c.get("section_heading", "")),
                score=c.get("rerank_score", c.get("score")),
                text=c.get("text", "")[:300],
            )
        )

    conf_score = result.get("retrieval_confidence", 0.0)
    confidence = ConfidenceBreakdown(
        retrieval_confidence=conf_score,
        composite=conf_score,
    )

    return AskResponse(
        query=result["query"],
        status=result["status"],
        answer=result["answer"],
        sources=sources,
        confidence=confidence,
    )


@app.get("/v1/documents", response_model=List[DocumentInfo], tags=["Documents"])
def get_documents() -> List[DocumentInfo]:
    """Lists all documents and chunk counts currently indexed in ChromaDB."""
    docs = chunking_mod.list_documents()
    return [DocumentInfo(**d) for d in docs]


@app.get("/v1/strategies", response_model=List[str], tags=["Documents"])
def get_strategies() -> List[str]:
    """Lists all distinct chunking strategies currently present in ChromaDB."""
    return chunking_mod.list_strategies()


@app.post("/v1/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest(req: IngestRequest) -> IngestResponse:
    """
    Ingests new chunks into ChromaDB and rebuilds the BM25 index:
    1. Embeds chunks with Ollama nomic-embed-text.
    2. Deduplicates against existing vectors (cosine similarity > 0.95).
    3. Upserts unique chunks to ChromaDB.
    4. Automatically resynchronizes BM25 sparse index so the two never diverge.
    """
    domain_chunks = [
        chunking_mod.Chunk(
            text=c.text,
            source_document=c.source_document,
            chunk_index=c.chunk_index,
            section_heading=c.section_heading,
            chunking_strategy=c.chunking_strategy,
        )
        for c in req.chunks
    ]

    duplicates = chunking_mod.upsert_to_chroma(domain_chunks)
    chunking_mod.rebuild_bm25()

    indexed_count = len(domain_chunks) - len(duplicates)
    return IngestResponse(
        status="success",
        chunks_submitted=len(domain_chunks),
        chunks_indexed=indexed_count,
        duplicates_skipped=len(duplicates),
    )


@app.post("/v1/upload", response_model=UploadDocumentResponse, tags=["Ingestion"])
async def upload_document(
    file: UploadFile = File(...),
    strategy: str = Form("recursive_character"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
) -> UploadDocumentResponse:
    """
    Accepts an uploaded file (PDF, TXT, MD, HTML), persists it to the pdfs/ folder,
    extracts pages/text, splits into chunks with metadata, embeds and upserts into ChromaDB,
    and resynchronizes the BM25 sparse index so it is immediately searchable.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ingest_mod.LOADER_MAPPING:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: {list(ingest_mod.LOADER_MAPPING.keys())}",
        )

    # Destination directory inside pdfs/
    save_dir = ROOT_DIR / "pdfs"
    save_dir.mkdir(parents=True, exist_ok=True)
    destination_path = save_dir / file.filename

    # Save uploaded bytes to disk
    try:
        with open(destination_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    finally:
        await file.close()

    # Parse document into LangChain documents
    try:
        docs = ingest_mod.load_document_from_file(destination_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse document: {e}")

    if not docs:
        raise HTTPException(status_code=400, detail="No readable content found in the uploaded document.")

    # Chunk with metadata
    chunks = splitter_mod.add_chunk_metadata(
        docs,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Convert to domain Chunk type for Chroma & BM25
    domain_chunks = [
        chunking_mod.Chunk(
            text=c.text,
            source_document=file.filename,
            chunk_index=c.chunk_index,
            section_heading=c.section_heading,
            chunking_strategy=c.chunking_strategy,
        )
        for c in chunks
    ]

    duplicates = chunking_mod.upsert_to_chroma(domain_chunks)
    chunking_mod.rebuild_bm25()

    try:
        total_indexed = chunking_mod.get_collection().count()
    except Exception:
        total_indexed = len(domain_chunks)

    return UploadDocumentResponse(
        status="success",
        filename=file.filename,
        pages_loaded=len(docs),
        chunks_created=len(domain_chunks),
        chunks_indexed=len(domain_chunks) - len(duplicates),
        duplicates_skipped=len(duplicates),
        total_chunks_indexed=total_indexed,
    )


@app.get("/health", tags=["Health"])
def health() -> dict:
    """Liveness check returning service status and total indexed chunk count."""
    try:
        col = chunking_mod.get_collection()
        total_chunks = col.count()
    except Exception:
        total_chunks = 0

    return {
        "status": "ok",
        "service": "Hybrid Search RAG API",
        "total_chunks_indexed": total_chunks,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
