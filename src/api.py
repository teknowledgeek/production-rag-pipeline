"""
api.py — FastAPI app exposing the RAG pipeline over HTTP.
 
Run locally:
    uvicorn src.api:app --reload
 
Docs UI:
    http://127.0.0.1:8000/docs
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


from src.retrieve import HybridRetriever
from src.generate import answer_question

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Loaded once at startup, reused across all requests — this is why we avoid
# reloading models per-request, which would be far too slow in production.
retriever: HybridRetriever | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    logger.info("Loading retriever (embedding model, re-ranker, BM25 index)...")
    retriever = HybridRetriever(chunks_path="./data/chunks.json")
    logger.info("Retriever ready. API is live.")
    yield
    logger.info("Shutting down.")

app = FastAPI(title="Production RAG Pipeline", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class Source(BaseModel):
    source: str
    page: int | None = None
    relevance_score: float

class QueryResponse(BaseModel):

    answer: str
    sources: list[Source]
    # sources: list[str] = []

@app.get("/")
def health_check():
    return {"status": "ok", "retriever loaded" : retriever is not None}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    # Placeholder for now — will call retrieve() + generate() here
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever is not initialized yet")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question can not be empty")

    try:
        result = answer_question(request.question, retriever, top_k=request.top_k)
    except Exception as e:
        logger.error(f"Error processing query {e}")
        raise(HTTPException(status_code=500, detail="Failed to generate an answer"))
    return QueryResponse(
        answer=result.answer,
        sources=result.sources
    )