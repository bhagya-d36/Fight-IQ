"""config.py — every tunable setting in one place, each overridable via an
environment variable (put overrides in .env; see .env.example for the list).

Values are read once at import time, so restart the app after changing them.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        sys.exit(f"Invalid {name}={raw!r}: expected a whole number.")


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        sys.exit(f"Invalid {name}={raw!r}: expected a number.")


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


LLM_PROVIDER = env_str("LLM_PROVIDER", "gemini").strip().lower()  # gemini, openai, anthropic, deepseek, kimi
CHAT_MODEL = env_str("CHAT_MODEL", "")  # blank = provider's default model (see llm.py)
EMBEDDING_MODEL = env_str("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # local sentence-transformers model
TOP_K = env_int("TOP_K", 4)  # how many chunks to retrieve per question
MIN_SIMILARITY = env_float("MIN_SIMILARITY", 0.35)  # below this, treat the KB as having no answer
MAX_CHUNK_CHARS = env_int("MAX_CHUNK_CHARS", 1500)  # ~350-400 tokens per chunk
LLM_TIMEOUT_MS = env_int("LLM_TIMEOUT_MS", 60_000)  # per-request timeout (per-read for streams)
LLM_RETRY_ATTEMPTS = env_int("LLM_RETRY_ATTEMPTS", 4)  # total tries per request, including the first
MAX_OUTPUT_TOKENS = env_int("MAX_OUTPUT_TOKENS", 1024)  # Anthropic requires an explicit cap
FORCE_IPV4 = env_bool("FORCE_IPV4")  # skip IPv6 DNS results; see net_fix.py
MAX_SESSIONS = env_int("MAX_SESSIONS", 20)  # web chat sessions kept in memory before LRU eviction
SESSION_TTL_MINUTES = env_int("SESSION_TTL_MINUTES", 60)  # idle web sessions expire after this
MAX_CHAT_TURNS = env_int("MAX_CHAT_TURNS", 20)  # sliding window of turns kept in a chat's history
ENABLE_QUERY_REWRITE = env_bool("ENABLE_QUERY_REWRITE", True)  # rewrite follow-ups into standalone search queries
REWRITE_HISTORY_TURNS = env_int("REWRITE_HISTORY_TURNS", 3)  # prior turns fed to the query rewriter
MAX_CHUNK_OVERLAP = env_int("MAX_CHUNK_OVERLAP", 200)  # chars re-included between packed chunks (re-ingest --force after changing)
HYBRID_SEARCH = env_bool("HYBRID_SEARCH", True)  # blend BM25 keyword search with vector search via RRF
RRF_K = env_int("RRF_K", 60)  # reciprocal-rank-fusion constant
MAX_QUESTION_CHARS = env_int("MAX_QUESTION_CHARS", 1000)  # reject questions longer than this
RATE_LIMIT_ENABLED = env_bool("RATE_LIMIT_ENABLED", True)  # cap requests per client IP on /api/ask*
RATE_LIMIT_REQUESTS = env_int("RATE_LIMIT_REQUESTS", 30)  # requests allowed per window, per client IP
RATE_LIMIT_WINDOW_SECONDS = env_int("RATE_LIMIT_WINDOW_SECONDS", 60)  # rate limit window length

