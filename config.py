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


CHAT_MODEL = env_str("CHAT_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = env_str("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = env_int("EMBEDDING_DIM", 768)  # plenty for a small KB; also the standard pgvector column size
TOP_K = env_int("TOP_K", 4)  # how many chunks to retrieve per question
MIN_SIMILARITY = env_float("MIN_SIMILARITY", 0.35)  # below this, treat the KB as having no answer
MAX_CHUNK_CHARS = env_int("MAX_CHUNK_CHARS", 1500)  # ~350-400 tokens per chunk
GEMINI_TIMEOUT_MS = env_int("GEMINI_TIMEOUT_MS", 60_000)  # per-request timeout (per-read for streams)
GEMINI_RETRY_ATTEMPTS = env_int("GEMINI_RETRY_ATTEMPTS", 4)  # total tries per request, including the first
FORCE_IPV4 = env_bool("FORCE_IPV4")  # skip IPv6 DNS results; see net_fix.py
