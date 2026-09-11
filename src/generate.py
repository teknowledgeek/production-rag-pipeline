"""
generate.py — Grounded generation: takes retrieved chunks + a question,
builds a citation-forcing prompt, and calls the LLM.

Usage (standalone test):
    python -m src.generate --query "your question here"

Used as a module by api.py.
"""

import argparse
import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from src.retrieve import HybridRetriever, RetrievedChunk

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Which LLM provider to use — "anthropic" or "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
ANTHROPIC_MODEL = "claude-sonnet-4-6"
OPENAI_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a precise research assistant. Answer the user's question \
using ONLY the provided context excerpts below. Follow these rules strictly:

1. Base your answer entirely on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, \
say "I don't have enough information in the provided documents to answer that" — \
do not guess or fill gaps with assumptions.
3. Cite which excerpt(s) support each claim using the format [Source: filename, page X].
4. Be concise and direct. Do not pad the answer with unnecessary caveats beyond \
what's stated above.
"""


@dataclass
class GenerationResult:
    answer: str
    sources: list[dict]


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Turn retrieved chunks into a numbered, citable context block for the prompt."""
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        page_info = f", page {chunk.page}" if chunk.page is not None else ""
        blocks.append(f"[Excerpt {i} — Source: {chunk.source}{page_info}]\n{chunk.text}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = format_context(chunks)
    return f"""Context excerpts:

{context}

---

Question: {question}

Answer using only the context above, with citations."""


def call_anthropic(system_prompt: str, user_prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


def call_openai(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1000,
    )
    return response.choices[0].message.content


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> GenerationResult:
    """Build the grounded prompt, call the configured LLM, and return answer + sources."""
    if not chunks:
        return GenerationResult(
            answer="I don't have enough information in the provided documents to answer that.",
            sources=[],
        )

    user_prompt = build_user_prompt(question, chunks)

    logger.info(f"Calling {LLM_PROVIDER} for generation...")
    if LLM_PROVIDER == "anthropic":
        answer = call_anthropic(SYSTEM_PROMPT, user_prompt)
    elif LLM_PROVIDER == "openai":
        answer = call_openai(SYSTEM_PROMPT, user_prompt)
    elif LLM_PROVIDER == "groq":
        answer = call_groq(SYSTEM_PROMPT, user_prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. Use 'anthropic' or 'openai'.")

    sources = [
        {"source": c.source, "page": c.page, "relevance_score": round(c.score, 4)}
        for c in chunks
    ]
    return GenerationResult(answer=answer, sources=sources)


def answer_question(question: str, retriever: HybridRetriever, top_k: int = 5) -> GenerationResult:
    """Convenience wrapper: retrieve + generate in one call. Used by api.py."""
    chunks = retriever.retrieve(question, top_k=top_k)
    return generate_answer(question, chunks)

def call_groq(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
    )
    # for m in client.models.list().data:
        # print(m.id)
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1000,
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="Test the full retrieve + generate pipeline.")
    parser.add_argument("--query", required=True, help="Question to ask")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunks", default="./data/chunks.json")
    args = parser.parse_args()

    retriever = HybridRetriever(chunks_path=args.chunks)
    result = answer_question(args.query, retriever, args.top_k)

    print(f"\nQuestion: {args.query}\n" + "-" * 60)
    print(f"\nAnswer:\n{result.answer}")
    print(f"\nSources used:")
    for s in result.sources:
        print(f"  - {s['source']} (page {s['page']}) — relevance: {s['relevance_score']}")


if __name__ == "__main__":
    main()