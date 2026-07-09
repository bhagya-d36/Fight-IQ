"""rag.py — shared retrieval + grounded-answer logic for chat.py and server.py.

Loads vector-store.json, embeds a question, retrieves the top-K most similar
chunks, and asks Gemini to answer using only that context.
"""

import json
import math
import os
from pathlib import Path

from google import genai
from google.genai import types

import config
from net_fix import prefer_ipv4

prefer_ipv4()

BASE_DIR = Path(__file__).resolve().parent
STORE_FILE = BASE_DIR / "vector-store.json"

# Tunables live in config.py (env-overridable); aliased here for convenience.
CHAT_MODEL = config.CHAT_MODEL
EMBEDDING_MODEL = config.EMBEDDING_MODEL
TOP_K = config.TOP_K
MIN_SIMILARITY = config.MIN_SIMILARITY

SYSTEM_INSTRUCTION = """You are a knowledgeable UFC/MMA assistant that answers questions using only the provided knowledge base.

Rules:
- Answer ONLY from the CONTEXT block provided with each question. It contains excerpts from the knowledge base.
- If the context does not contain the answer, say you don't have that information. Never guess.
- Quote records, dates, weights, and other figures exactly as written in the context. Never invent, estimate, or round them.
- Never state a fighter's ranking, title status, or fight result unless it's stated in the context — these change often.
- Be concise and clear."""


def load_store() -> dict:
    if not STORE_FILE.exists():
        raise FileNotFoundError("vector-store.json not found. Run `python ingest.py` first.")
    return json.loads(STORE_FILE.read_text(encoding="utf-8"))


def make_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY. Create a .env file (see .env.example).")
    return genai.Client(api_key=api_key)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def retrieve(
    store: dict,
    client: genai.Client,
    question: str,
    top_k: int = TOP_K,
    min_similarity: float = MIN_SIMILARITY,
) -> list[dict]:
    res = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[question],
        config=types.EmbedContentConfig(
            output_dimensionality=store["dim"],
            task_type="RETRIEVAL_QUERY",
        ),
    )
    q_vec = list(res.embeddings[0].values)
    scored = [
        {**entry, "score": cosine_similarity(q_vec, entry["embedding"])}
        for entry in store["entries"]
    ]
    scored.sort(key=lambda e: e["score"], reverse=True)
    return [e for e in scored[:top_k] if e["score"] >= min_similarity]


def build_context(hits: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[{i + 1}] (source: {h['source']})\n{h['text']}" for i, h in enumerate(hits)
    )


def answer(client: genai.Client, question: str, hits: list[dict]) -> str:
    """Single-turn grounded answer (no shared chat history across callers)."""
    context = build_context(hits)
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=f"CONTEXT:\n{context}\n\nQUESTION: {question}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
        ),
    )
    return response.text


def answer_stream(client: genai.Client, question: str, hits: list[dict]):
    """Yields response text chunks for a single-turn grounded answer."""
    context = build_context(hits)
    stream = client.models.generate_content_stream(
        model=CHAT_MODEL,
        contents=f"CONTEXT:\n{context}\n\nQUESTION: {question}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
        ),
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text
