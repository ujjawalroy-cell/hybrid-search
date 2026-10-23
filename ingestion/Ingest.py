from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader, 
)
import glob
from pathlib import Path
from typing import List, Union
from langchain_core.documents import Document

DATA_DIR = Path("C:/ujjawal/projects/Hybrid_search_with_RAG/pdfs")

# Map file extensions to their corresponding LangChain loader classes
LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
    ".md": UnstructuredMarkdownLoader,
    ".txt": TextLoader,}


def load_documents_from_directory(directory: Path) -> List[Document]:
    """Scans directory for supported files and returns a list of LangChain Documents."""
    documents = []

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Iterate over every file in the directory
    for file_path in directory.rglob("*"):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        loader_cls = LOADER_MAPPING.get(ext)

        if loader_cls:
            try:
                # Instantiates loader with a single file path string
                loader = loader_cls(str(file_path))
                docs = loader.load()

                # Optional: Attach file extension metadata
                for doc in docs:
                    doc.metadata["file_type"] = ext

                documents.extend(docs)
                print(f"Loaded {len(docs)} document(s) from {file_path.name}")

            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")

def load_document_from_file(file_path: Union[str, Path]) -> List[Document]:
    """Loads a single file (PDF, TXT, MD, HTML) and returns a list of LangChain Documents."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    ext = p.suffix.lower()
    loader_cls = LOADER_MAPPING.get(ext)
    if not loader_cls:
        raise ValueError(f"Unsupported file type '{ext}'. Supported types: {list(LOADER_MAPPING.keys())}")

    try:
        loader = loader_cls(str(p))
        docs = loader.load()
        for doc in docs:
            doc.metadata["file_type"] = ext
            doc.metadata["source"] = p.name
        return docs
    except Exception as e:
        print(f"Error loading {p.name}: {e}")
        raise e


if __name__ == "__main__":
    all_docs = load_documents_from_directory(DATA_DIR)
    print(f"\nTotal loaded document chunks/pages: {len(all_docs)}")