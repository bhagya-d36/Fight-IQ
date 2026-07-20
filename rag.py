"""rag.py — shared retrieval + grounded-answer logic for chat.py and server.py.

Opens the local Chroma vector store, embeds a question, retrieves the top-K
most similar chunks, and asks the configured chat LLM to answer using only
that context.
"""

import math
import re
from collections.abc import Iterator
from pathlib import Path

import chromadb

import config
import embeddings
from net_fix import prefer_ipv4

prefer_ipv4()

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma-store"
COLLECTION_NAME = "chunks"

# How many candidates the ANN index returns for hybrid re-ranking. Must be
# wide enough that a strong keyword match outside the top cosine hits still
# gets pulled into the RRF fusion below.
DENSE_CANDIDATE_POOL = 100

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
- Never state a fighter's ranking, title status, or fight result unless it's stated in the context — these change often. When you do state one, note that it reflects the knowledge base's last update and may no longer be current.
- When a fact comes from a specific context excerpt, briefly note its source (e.g., "per rankings.md").
- Each CONTEXT excerpt is numbered like [1], [2]. When you state a fact, cite the excerpt it came from using its bracket number (e.g. "Aspinall is champion [1]."). Cite multiple as [2][3].
- Be concise and clear."""

NO_MATCH_ANSWER = "I don't have information about that in my knowledge base."
EMPTY_RESPONSE_ANSWER = "The model returned an empty response. Please try asking again."

REWRITE_INSTRUCTION = (
    "You rewrite a user's follow-up question into a standalone search query.\n"
    "Use the conversation to resolve pronouns and references (e.g. 'he', 'that fight', 'the champ').\n"
    "If the question is already standalone, return it unchanged.\n"
    "Output ONLY the rewritten query — no preamble, no quotes, no explanation."
)

BM25_K1 = 1.5
BM25_B = 0.75
_TOKEN_RE = re.compile(r"[0-9]+(?:-[0-9]+)+|\w+", re.UNICODE)


def validate_store(store: dict) -> None:
    """Raise ValueError with an actionable message if the store looks unusable."""
    entries = store.get("entries")
    if not entries:
        raise ValueError("Chroma store has no entries. Run `python ingest.py --force`.")
    if store.get("model") != config.EMBEDDING_MODEL or store.get("dim") != embeddings.dimension():
        raise ValueError(
            f"Chroma store was built with model={store.get('model')!r} "
            f"dim={store.get('dim')!r}, but the app is configured for "
            f"model={config.EMBEDDING_MODEL!r} dim={embeddings.dimension()!r}. "
            "Run `python ingest.py --force` to rebuild it."
        )


def load_store() -> dict:
    """Open the Chroma collection and pull chunk text + metadata into memory
    for BM25 (embeddings stay in Chroma — that's what keeps this cheap)."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except chromadb.errors.NotFoundError as err:
        raise FileNotFoundError("Chroma store not found. Run `python ingest.py` first.") from err

    meta = collection.metadata or {}
    got = collection.get(include=["documents", "metadatas"])
    entries = [{"source": m["source"], "text": d} for m, d in zip(got["metadatas"], got["documents"])]

    store = {
        "collection": collection,
        "entries": entries,
        "ids": got["ids"],
        "_id_to_idx": {cid: i for i, cid in enumerate(got["ids"])},
        "model": meta.get("model"),
        "dim": meta.get("dim"),
        "version": meta.get("version"),
    }
    validate_store(store)
    return store


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _bm25_index(store: dict) -> dict:
    """Build (and cache on the store dict) BM25 corpus stats: per-doc term
    frequencies, inverse doc frequencies, doc lengths, and average length.
    """
    idx = store.get("_bm25")
    if idx is not None:
        return idx
    docs = [_tokenize(e["text"]) for e in store["entries"]]
    doc_freq: dict[str, int] = {}
    for toks in docs:
        for t in set(toks):
            doc_freq[t] = doc_freq.get(t, 0) + 1
    n = len(docs)
    avgdl = (sum(len(d) for d in docs) / n) if n else 0.0
    idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in doc_freq.items()}
    term_freq: list[dict[str, int]] = [{} for _ in docs]
    for i, toks in enumerate(docs):
        for t in toks:
            term_freq[i][t] = term_freq[i].get(t, 0) + 1
    idx = {"tf": term_freq, "idf": idf, "len": [len(d) for d in docs], "avgdl": avgdl}
    store["_bm25"] = idx
    return idx


def _bm25_scores(store: dict, query: str) -> list[float]:
    idx = _bm25_index(store)
    q_tokens = _tokenize(query)
    scores = [0.0] * len(idx["len"])
    for i in range(len(scores)):
        doc_len, tf = idx["len"][i], idx["tf"][i]
        s = 0.0
        for t in q_tokens:
            f = tf.get(t)
            if not f:
                continue
            denom = f + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / (idx["avgdl"] or 1))
            s += idx["idf"].get(t, 0.0) * f * (BM25_K1 + 1) / denom
        scores[i] = s
    return scores


def _ranks(scores: list[float]) -> list[int]:
    """1-based rank of each score, best (highest) first; ties broken by index."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    rank = [0] * len(scores)
    for pos, i in enumerate(order, start=1):
        rank[i] = pos
    return rank


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def _dense_pool(store: dict, q_vec: list[float], pool_size: int) -> tuple[list[str], list[dict], list[float]]:
    """Top `pool_size` chunks by cosine similarity, via Chroma's HNSW index.
    Returns (chunk ids, entries, cosine scores), descending by similarity.

    Chroma is built with hnsw:space="cosine", so it returns *distance*
    (1 - cosine similarity), not similarity — converted back here.
    """
    res = store["collection"].query(
        query_embeddings=[q_vec], n_results=pool_size, include=["documents", "metadatas", "distances"]
    )
    ids = res["ids"][0]
    pool_entries = [{"source": m["source"], "text": d} for m, d in zip(res["metadatas"][0], res["documents"][0])]
    pool_cos = [1.0 - dist for dist in res["distances"][0]]
    return ids, pool_entries, pool_cos


def retrieve(
    store: dict,
    question: str,
    top_k: int = TOP_K,
    min_similarity: float = MIN_SIMILARITY,
    hybrid: bool | None = None,
) -> list[dict]:
    if hybrid is None:
        hybrid = config.HYBRID_SEARCH

    n = len(store["entries"])
    if n == 0:
        return []

    q_vec = embeddings.embed_texts([question])[0]
    pool_size = min(max(top_k * 10, DENSE_CANDIDATE_POOL), n)
    pool_ids, pool_entries, pool_cos = _dense_pool(store, q_vec, pool_size)

    def top_by_cosine() -> list[dict]:
        scored = [{**e, "score": c} for e, c in zip(pool_entries, pool_cos)]
        scored.sort(key=lambda e: e["score"], reverse=True)
        return [e for e in scored[:top_k] if e["score"] >= min_similarity]

    if not hybrid:
        return top_by_cosine()

    bm25 = _bm25_scores(store, question)
    if not any(s > 0 for s in bm25):
        return top_by_cosine()

    # Grounding gate stays cosine-only: keyword overlap alone can never
    # rescue a question the KB has no real answer for.
    if (max(pool_cos) if pool_cos else 0.0) < min_similarity:
        return []

    # RRF fuses the dense candidate pool with every keyword-matching entry
    # (BM25 is cheap over the full corpus; the ANN pool is not exhaustive).
    id_to_idx = store["_id_to_idx"]
    pool_idx = [id_to_idx[cid] for cid in pool_ids]
    cos_by_idx = dict(zip(pool_idx, pool_cos))
    kw_idx = [i for i, s in enumerate(bm25) if s > 0]
    candidate_idx = list(dict.fromkeys(pool_idx + kw_idx))  # ordered dedup

    # A keyword-only candidate (outside the dense pool) has no cosine score
    # yet — fetch its embedding directly so the displayed score is always a
    # real similarity, never a rank-only placeholder.
    missing_idx = [i for i in candidate_idx if i not in cos_by_idx]
    if missing_idx:
        ids = store["ids"]
        got = store["collection"].get(ids=[ids[i] for i in missing_idx], include=["embeddings"])
        emb_by_id = dict(zip(got["ids"], got["embeddings"]))
        for i in missing_idx:
            cos_by_idx[i] = _cosine_similarity(q_vec, emb_by_id[ids[i]])

    cand_cos = [cos_by_idx[i] for i in candidate_idx]
    cand_bm25 = [bm25[i] for i in candidate_idx]
    cos_rank, kw_rank = _ranks(cand_cos), _ranks(cand_bm25)

    def rrf(pos: int) -> float:
        r = 1.0 / (config.RRF_K + cos_rank[pos])
        if cand_bm25[pos] > 0:
            r += 1.0 / (config.RRF_K + kw_rank[pos])
        return r

    order = sorted(range(len(candidate_idx)), key=lambda p: (-rrf(p), -cand_cos[p], p))
    entries = store["entries"]
    return [{**entries[candidate_idx[p]], "score": cand_cos[p]} for p in order[:top_k]]


def build_context(hits: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[{i + 1}] (source: {h['source']})\n{h['text']}" for i, h in enumerate(hits)
    )


def build_prompt(question: str, hits: list[dict]) -> str:
    return f"CONTEXT:\n{build_context(hits)}\n\nQUESTION: {question}"


def rewrite_query(provider, question: str, turns: list[dict]) -> str:
    """Rewrite a follow-up into a standalone search query using recent (q, a)
    turns, so retrieval can resolve pronouns like "he" or "that fight".
    Falls back to the original question on any error.
    """
    convo = "\n".join(f"User: {t['q']}\nAssistant: {t['a']}" for t in turns)
    prompt = f"{REWRITE_INSTRUCTION}\n\nCONVERSATION:\n{convo}\n\nFOLLOW-UP: {question}\n\nSTANDALONE QUERY:"
    try:
        rewritten = provider.complete(prompt, temperature=0.0).strip().strip('"').strip()
        return rewritten or question
    except Exception:
        return question


class GroundedChat:
    """Multi-turn grounded chat: per-question retrieval plus a locally-kept
    (question, answer) history so follow-ups ("and his last fight?") resolve.
    Only the bare Q&A is stored per turn — retrieved context is injected only
    for the current question, never resent for prior turns.
    """

    def __init__(self, store: dict, provider, max_turns: int = config.MAX_CHAT_TURNS) -> None:
        self._store = store
        self._provider = provider
        self._max_turns = max_turns
        self._turns: list[dict] = []

    def _history_messages(self) -> list[dict]:
        messages = []
        for t in self._turns:
            messages.append({"role": "user", "content": t["q"]})
            messages.append({"role": "assistant", "content": t["a"]})
        return messages

    def _search_query(self, question: str) -> str:
        if config.ENABLE_QUERY_REWRITE and self._turns:
            recent = self._turns[-config.REWRITE_HISTORY_TURNS :]
            return rewrite_query(self._provider, question, recent)
        return question

    def _record_turn(self, question: str, answer: str) -> None:
        self._turns.append({"q": question, "a": answer})
        self._turns = self._turns[-self._max_turns :]

    def ask(self, question: str) -> tuple[str, list[dict]]:
        search_query = self._search_query(question)
        hits = retrieve(self._store, search_query)
        if not hits:
            return NO_MATCH_ANSWER, hits
        messages = self._history_messages() + [{"role": "user", "content": build_prompt(question, hits)}]
        answer = self._provider.chat(messages, system=SYSTEM_INSTRUCTION, temperature=0.2)
        if answer:
            self._record_turn(question, answer)
        return answer or EMPTY_RESPONSE_ANSWER, hits

    def ask_stream(self, question: str) -> tuple[list[dict], Iterator[str]]:
        search_query = self._search_query(question)
        hits = retrieve(self._store, search_query)
        if not hits:
            return hits, iter((NO_MATCH_ANSWER,))
        messages = self._history_messages() + [{"role": "user", "content": build_prompt(question, hits)}]
        stream = self._provider.stream_chat(messages, system=SYSTEM_INSTRUCTION, temperature=0.2)

        def chunks() -> Iterator[str]:
            parts: list[str] = []
            for piece in stream:
                if piece:
                    parts.append(piece)
                    yield piece
            answer = "".join(parts)
            if answer:
                self._record_turn(question, answer)

        return hits, chunks()


