from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Production RAG Pipeline")


class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):

    answer: str
    sources: list[str] = []

@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    # Placeholder for now — will call retrieve() + generate() here
    return QueryResponse(
        answer=f"You asked: {request.question} (RAG Logic not wired up yet)",
        sources=[]
    )