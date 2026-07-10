"""rag.py — shared retrieval + grounded-answer logic for chat.py and server.py.

Loads vector-store.json, embeds a question, retrieves the top-K most similar
chunks, and asks Gemini to answer using only that context.
"""

import json
import math
import os
from collections.abc import Iterator
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

NO_MATCH_ANSWER = "I don't have information about that in my knowledge base."
EMPTY_RESPONSE_ANSWER = "The model returned an empty response. Please try asking again."


def validate_store(store: dict) -> None:
    """Raise ValueError with an actionable message if the store looks unusable."""
    entries = store.get("entries")
    if not entries:
        raise ValueError("vector-store.json has no entries. Run `python ingest.py --force`.")
    if store.get("model") != config.EMBEDDING_MODEL or store.get("dim") != config.EMBEDDING_DIM:
        raise ValueError(
            f"vector-store.json was built with model={store.get('model')!r} "
            f"dim={store.get('dim')!r}, but the app is configured for "
            f"model={config.EMBEDDING_MODEL!r} dim={config.EMBEDDING_DIM!r}. "
            "Run `python ingest.py --force` to rebuild it."
        )
    if len(entries[0].get("embedding", [])) != store["dim"]:
        raise ValueError(
            "vector-store.json entries don't match its declared dim. "
            "Run `python ingest.py --force` to rebuild it."
        )


def load_store() -> dict:
    if not STORE_FILE.exists():
        raise FileNotFoundError("vector-store.json not found. Run `python ingest.py` first.")
    try:
        store = json.loads(STORE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ValueError("vector-store.json is corrupt. Run `python ingest.py --force`.") from err
    validate_store(store)
    return store


def make_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY. Create a .env file (see .env.example).")
    # The SDK retries 408/429/5xx and connect/timeout errors with exponential
    # backoff. Only the initial request is retried for streams — a drop
    # mid-stream is not.
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=config.GEMINI_TIMEOUT_MS,  # milliseconds; per-read for streams
            retry_options=types.HttpRetryOptions(attempts=config.GEMINI_RETRY_ATTEMPTS),
        ),
    )


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


def build_prompt(question: str, hits: list[dict]) -> str:
    return f"CONTEXT:\n{build_context(hits)}\n\nQUESTION: {question}"


class GroundedChat:
    """Multi-turn grounded chat: per-question retrieval plus a Gemini chat
    session that keeps history so follow-ups ("and his last fight?") resolve.
    """

    def __init__(self, store: dict, client: genai.Client, max_turns: int = config.MAX_CHAT_TURNS) -> None:
        self._store = store
        self._client = client
        self._max_turns = max_turns
        self._chat = self._new_chat()

    def _new_chat(self, history=None):
        return self._client.chats.create(
            model=CHAT_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,
            ),
            history=history,
        )

    def _trim_history(self) -> None:
        history = self._chat.get_history(curated=True)
        limit = self._max_turns * 2
        if len(history) > limit:
            self._chat = self._new_chat(history=history[-limit:])

    def ask(self, question: str) -> tuple[str, list[dict]]:
        hits = retrieve(self._store, self._client, question)
        if not hits:
            return NO_MATCH_ANSWER, hits
        self._trim_history()
        response = self._chat.send_message(build_prompt(question, hits))
        return response.text or EMPTY_RESPONSE_ANSWER, hits

    def ask_stream(self, question: str) -> tuple[list[dict], Iterator[str]]:
        hits = retrieve(self._store, self._client, question)
        if not hits:
            return hits, iter((NO_MATCH_ANSWER,))
        self._trim_history()
        stream = self._chat.send_message_stream(build_prompt(question, hits))

        def chunks() -> Iterator[str]:
            for chunk in stream:
                if chunk.text:
                    yield chunk.text

        return hits, chunks()
