"""
retrieve.py — Hybrid retrieval (dense + BM25) with cross-encoder re-ranking.

Usage (standalone test):
    python -m src.retrieve --query "your question here" --top-k 5

Used as a module by generate.py / api.py via the retrieve() function.
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "rag_docs"


# Re-ranking adds real retrieval quality but also loads a second transformer
# model into memory. On memory-constrained deployments (e.g. Render's free
# 512MB tier), set ENABLE_RERANKING=false to skip loading it entirely.
# This is a deliberate, documented trade-off — see README for eval comparison.
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"

# How many candidates each retrieval method pulls before fusion/re-ranking.
# Wider than final top_k so re-ranking has real signal to work with.
CANDIDATE_POOL_SIZE = 20


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int | None
    score: float


class HybridRetriever:
    """
    Combines dense vector search (Qdrant) with sparse BM25 keyword search,
    fuses the results, and re-ranks the merged pool with a cross-encoder.
    """

    def __init__(
        self,
        chunks_path: str = "./data/chunks.json",
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        reranker_model: str = RERANKER_MODEL,
    ):
        logger.info("Loading chunks for BM25 index...")
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # BM25 needs tokenized text — simple whitespace tokenization is fine here
        tokenized_corpus = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

        if ENABLE_RERANKING:
            logger.info(f"Loading re-ranker: {reranker_model}")
            self.reranker = CrossEncoder(reranker_model)
        else:
            logger.info("Re-ranking disabled (ENABLE_RERANKING=false) — skipping reranker load to save memory")
            self.reranker = None

        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY")
        self.qdrant = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name

        logger.info("HybridRetriever ready.")

    def _dense_search(self, query: str, k: int) -> list[dict]:
        """Vector similarity search against Qdrant."""
        query_vector = self.embedder.encode(query, normalize_embeddings=True).tolist()
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
        ).points
        return [
            {
                "text": r.payload["text"],
                "source": r.payload["source"],
                "page": r.payload.get("page"),
                "score": r.score,
            }
            for r in results
        ]

    def _bm25_search(self, query: str, k: int) -> list[dict]:
        """Sparse keyword search using BM25 over the full chunk corpus."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[::-1][:k]
        return [
            {
                "text": self.chunks[i]["text"],
                "source": self.chunks[i]["source"],
                "page": self.chunks[i].get("page"),
                "score": float(scores[i]),
            }
            for i in top_indices
            if scores[i] > 0  # skip zero-relevance BM25 matches
        ]

    def _fuse_results(self, dense: list[dict], sparse: list[dict]) -> list[dict]:
        """
        Merge dense + sparse results, deduplicating by text.
        Uses Reciprocal Rank Fusion (RRF) — combines rankings rather than
        raw scores, since dense (cosine) and BM25 scores aren't on the same scale.
        """
        rrf_k = 60  # standard RRF constant
        fused_scores: dict[str, float] = {}
        chunk_lookup: dict[str, dict] = {}

        for rank, item in enumerate(dense):
            key = item["text"]
            fused_scores[key] = fused_scores.get(key, 0) + 1 / (rrf_k + rank + 1)
            chunk_lookup[key] = item

        for rank, item in enumerate(sparse):
            key = item["text"]
            fused_scores[key] = fused_scores.get(key, 0) + 1 / (rrf_k + rank + 1)
            chunk_lookup.setdefault(key, item)

        ranked_keys = sorted(fused_scores, key=lambda k: fused_scores[k], reverse=True)
        return [chunk_lookup[k] for k in ranked_keys]

    def _rerank(self, query: str, candidates: list[dict], top_k: int) -> list[RetrievedChunk]:
        """
        Re-score the fused candidate pool with a cross-encoder for final ranking.
        If reranking is disabled, falls back to the fused (RRF) order as-is —
        still a real hybrid result, just without the cross-encoder's extra precision.
        """
        if not candidates:
            return []

        if self.reranker is None:
            return [
                RetrievedChunk(
                    text=c["text"],
                    source=c["source"],
                    page=c.get("page"),
                    score=c.get("score", 0.0),
                )
                for c in candidates[:top_k]
            ]

        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievedChunk(
                text=c["text"],
                source=c["source"],
                page=c.get("page"),
                score=float(score),
            )
            for c, score in scored[:top_k]
        ]

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Full pipeline: dense + sparse retrieval → fusion → cross-encoder re-rank."""
        dense_results = self._dense_search(query, CANDIDATE_POOL_SIZE)
        sparse_results = self._bm25_search(query, CANDIDATE_POOL_SIZE)

        fused = self._fuse_results(dense_results, sparse_results)
        reranked = self._rerank(query, fused, top_k)

        logger.info(
            f"Retrieved {len(dense_results)} dense + {len(sparse_results)} sparse "
            f"→ fused to {len(fused)} → re-ranked to top {len(reranked)}"
        )
        return reranked


def main():
    parser = argparse.ArgumentParser(description="Test hybrid retrieval against the indexed corpus.")
    parser.add_argument("--query", required=True, help="Question to search for")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunks", default="./data/chunks.json")
    args = parser.parse_args()

    retriever = HybridRetriever(chunks_path=args.chunks)
    results = retriever.retrieve(args.query, args.top_k)

    print(f"\nTop {len(results)} results for: \"{args.query}\"\n" + "-" * 60)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] score={r.score:.4f} | source={r.source} | page={r.page}")
        print(r.text[:300] + ("..." if len(r.text) > 300 else ""))


if __name__ == "__main__":
    main()