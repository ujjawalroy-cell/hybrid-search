from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)
@dataclass
class Chunk:
    text: str
    source_document: str
    chunk_index: int
    section_heading: str
    chunking_strategy: str
 
    @property
    def id(self) -> str:
        # Deterministic ID: re-running the pipeline upserts, never duplicates.
        return f"{self.source_document}::{self.chunking_strategy}::{self.chunk_index}"
 
    @property
    def metadata(self) -> dict:
        # Chroma metadata values must be str/int/float/bool (no None).
        return {
            "source_document": self.source_document,
            "chunk_index": self.chunk_index,
            "section_heading": self.section_heading or "",
            "chunking_strategy": self.chunking_strategy,
            "char_count": len(self.text),
        }
 

def get_text_splitter(strategy: str = "recursive_character", chunk_size: int = 1000,chunk_overlap: int = 200):
    """
    Factory function returning the configured text splitter based on strategy.
    """
    strategy = strategy.lower().strip()
    if strategy in ("recursive_character", "recursive"):
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
    elif strategy == "character":
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n",
        )
    elif strategy == "token":
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        # Default fallback
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def add_chunk_metadata(
    docs: List[Document],
    strategy: str = "recursive_character",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Chunk]:
    """
    Splits LangChain documents into chunks and wraps each into a typed
    Chunk object with clean metadata (source document, index, heading, strategy)
    ready for Chroma upsert and BM25 indexing.
    """
    splitter = get_text_splitter(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = splitter.split_documents(docs)  # data is divided into chunks

    chunks: List[Chunk] = []   
    doc_chunk_counters: dict[str, int] = {}   # for count of chunks per document
    for split_doc in split_docs:
        text = split_doc.page_content.strip() #remove leading/trailing whitespace from page_content

        if not text: #skip if the chunk is empty
            continue
        raw_source = split_doc.metadata.get("source", "unknown") # Get the source document
        
        source_name = Path(raw_source).name if raw_source != "unknown" else "unknown" # Normalize to filename so IDs remain clean and platform-independent

        curr_index = doc_chunk_counters.get(source_name, 0) # Get the current chunk index for the source document
        doc_chunk_counters[source_name] = curr_index + 1 # Increment the chunk index for the source document

        # Determine section or heading if available
        section = (
            split_doc.metadata.get("section_heading")
            or split_doc.metadata.get("section")
            or split_doc.metadata.get("title")
        )
        if not section: # If no section is found, try to get the page number
            page = split_doc.metadata.get("page")
            if page is not None:
                section = f"Page {page + 1}"
            else:
                section = ""

        chunk = Chunk(
            text=text,
            source_document=source_name,
            chunk_index=curr_index,
            section_heading=str(section),
            chunking_strategy=strategy,
        )
        chunks.append(chunk)

    print(doc_chunk_counters)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks using strategy '{strategy}'.")
    return chunks