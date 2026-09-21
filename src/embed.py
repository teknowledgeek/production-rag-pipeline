"""
embed.py — Embed chunks and index them into Qdrant.
 
Supports two embedding backends, switchable via EMBEDDING_BACKEND env var:
  - "fastembed" (default): ONNX-quantized, no PyTorch dependency, ~100MB
    memory footprint. Use this for memory-constrained deployments
    (Render free tier, etc.).
  - "sentence-transformers": full PyTorch backend, more model choice and
    ecosystem support, but ~300-500MB+ heavier. Fine for local dev or
    deployments with more RAM (Hugging Face Spaces Pro, etc.).
 
Usage:
    python -m src.embed --input ./data/chunks.json --collection rag_docs
 
Requires a running Qdrant instance:
    docker run -p 6333:6333 qdrant/qdrant
Or Qdrant Cloud — set QDRANT_URL / QDRANT_API_KEY in .env.
"""

import argparse
import json
import logging
import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
# from sentence_transformers import SentenceTransformer

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Which backend to use — "fastembed" (default, lightweight) or "sentence-transformers"
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "fastembed").lower()

# Default: open-source embedding model, runs locally, no API cost.
# Swap to "text-embedding-3-small" + OpenAI client if you'd rather use OpenAI.
# Model names per backend — both map to the same underlying BGE-small model,
# so embeddings are conceptually equivalent, though not guaranteed bit-identical.
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
SENTENCE_TRANSFORMERS_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # bge-small output dimension — same for both backends


class EmbeddingModel:
    """
    Thin wrapper unifying fastembed and sentence-transformers behind one
    interface, so the rest of the pipeline doesn't care which is active.
    """
    def __init__(self, backend: str = EMBEDDING_BACKEND):
        self.backend = backend

        if backend == "fastembed":
            from fastembed import TextEmbedding

            logger.info(f"Loading fastembed model: {FASTEMBED_MODEL} (lightweight, ONNX-based)")
            self._model = TextEmbedding(model_name=FASTEMBED_MODEL)
 
        elif backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence-transformers model: {SENTENCE_TRANSFORMERS_MODEL} (full PyTorch)")
            self._model = SentenceTransformer(SENTENCE_TRANSFORMERS_MODEL)
 
        else:
            raise ValueError(
                f"Unknown EMBEDDING_BACKEND: {backend}. Use 'fastembed' or 'sentence-transformers'."
            )

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts into normalized embedding vectors. Returns list of lists."""
        if self.backend == "fastembed":
            # fastembed.embed() returns a generator of numpy arrays, already normalized by default
            embeddings = list(self._model.embed(texts))
            return [e.tolist() for e in embeddings]
 
        else:  # sentence-transformers
            embeddings = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            return [e.tolist() for e in embeddings]
 
    def get_dimension(self) -> int:
        return EMBEDDING_DIM


def load_chunks(input_path: str) -> list[dict]:
    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks from {input_path}")
    return chunks


# def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
#     logger.info(f"Loading embedding model: {model_name} (first run downloads it, ~130MB)")
#     return SentenceTransformer(model_name)


def get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")  # None for local, required for Qdrant Cloud
    logger.info(f"Connecting to Qdrant at {url}")
    return QdrantClient(url=url, api_key=api_key)


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        logger.info(f"Collection '{collection_name}' already exists — reusing it")
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    logger.info(f"Created collection '{collection_name}' (dim={vector_size}, cosine distance)")


def embed_and_index(
    chunks: list[dict],
    model: EmbeddingModel,
    client: QdrantClient,
    collection_name: str,
    batch_size: int = 64,
) -> None:
    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start:start + batch_size]
        texts = [c["text"] for c in batch]
 
        vectors = model.encode(texts)
 
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "page": chunk.get("page"),
                    "chunk_id": chunk["id"],
                },
            )
            for chunk, vector in zip(batch, vectors)
        ]
 
        client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Indexed {min(start + batch_size, total)}/{total} chunks")
 
    logger.info(f"Finished indexing {total} chunks into '{collection_name}' using {model.backend} backend")



def main():
    parser = argparse.ArgumentParser(description="Embed chunks and index into Qdrant.")
    parser.add_argument("--input", default="./data/chunks.json", help="Path to chunks.json from ingest.py")
    parser.add_argument("--collection", default="rag_docs", help="Qdrant collection name")
    parser.add_argument(
        "--backend",
        default=EMBEDDING_BACKEND,
        choices=["fastembed", "sentence-transformers"],
        help="Embedding backend to use",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
 
    chunks = load_chunks(args.input)
    if not chunks:
        logger.error("No chunks found. Run ingest.py first.")
        return
 
    model = EmbeddingModel(backend=args.backend)
 
    client = get_qdrant_client()
    ensure_collection(client, args.collection, model.get_dimension())
    embed_and_index(chunks, model, client, args.collection, args.batch_size)
 
 
if __name__ == "__main__":
    main()
 