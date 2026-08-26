"""
ingest.py — Document loading, cleaning, and chunking for the RAG pipeline.

Usage:
    python -m src.ingest --source ./data/docs --output ./data/chunks.json
"""

import argparse
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Map file extensions to their loader
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".md": UnstructuredMarkdownLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
    ".txt": TextLoader,
}


@dataclass
class Chunk:
    """A single chunk of text with metadata, ready for embedding."""
    id: str
    text: str
    source: str
    page: int | None = None
    chunk_index: int = 0


def load_documents(source_dir: str) -> list:
    """
    Walk a directory and load every supported file using the right loader.
    Returns a list of LangChain Document objects with metadata attached.
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    documents = []
    files = [f for f in source_path.rglob("*") if f.suffix.lower() in LOADER_MAP]

    if not files:
        logger.warning(f"No supported files found in {source_dir}. "
                        f"Supported types: {list(LOADER_MAP.keys())}")
        return documents

    for file_path in files:
        loader_cls = LOADER_MAP[file_path.suffix.lower()]
        try:
            loader = loader_cls(str(file_path))
            docs = loader.load()
            for doc in docs:
                # Normalize metadata across loader types
                doc.metadata["source"] = str(file_path.relative_to(source_path))
            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} section(s) from {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


def clean_text(text: str) -> str:
    """Basic text normalization — strip excessive whitespace and empty lines."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def chunk_documents(
    documents: list,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Split loaded documents into overlapping chunks using a recursive splitter,
    which respects paragraph/sentence boundaries before falling back to hard splits.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for doc_idx, doc in enumerate(documents):
        cleaned = clean_text(doc.page_content)
        if not cleaned:
            continue

        splits = splitter.split_text(cleaned)
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")

        for i, split_text in enumerate(splits):
            chunk = Chunk(
                id=f"{source}::{doc_idx}::{i}",
                text=split_text,
                source=source,
                page=page,
                chunk_index=i,
            )
            chunks.append(chunk)

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def save_chunks(chunks: list[Chunk], output_path: str) -> None:
    """Save chunks to a JSON file for the embedding step to consume."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Ingest and chunk documents for RAG.")
    parser.add_argument("--source", required=True, help="Directory containing source documents")
    parser.add_argument("--output", default="./data/chunks.json", help="Where to save chunked output")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    args = parser.parse_args()

    documents = load_documents(args.source)
    if not documents:
        logger.error("No documents were loaded. Exiting.")
        return

    chunks = chunk_documents(documents, args.chunk_size, args.chunk_overlap)
    save_chunks(chunks, args.output)


if __name__ == "__main__":
    main()