"""sessions.py — bounded in-memory session store for the web UI.

Keeps at most `max_sessions` values alive, evicting the least-recently-used
entry when full, and treats entries idle past `ttl_seconds` as expired.
Unknown or expired session ids transparently get a fresh value rather than
raising, since callers (the web endpoints) can't tell a stale id from a new
browser tab.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class _Entry(Generic[T]):
    value: T
    last_used: float


class SessionStore(Generic[T]):
    def __init__(
        self,
        factory: Callable[[], T],
        max_sessions: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._factory = factory
        self._max_sessions = max_sessions
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: dict[str, _Entry[T]] = {}
        self._lock = threading.Lock()

    def _evict_expired(self, now: float) -> None:
        expired = [
            sid for sid, entry in self._entries.items() if now - entry.last_used > self._ttl_seconds
        ]
        for sid in expired:
            del self._entries[sid]

    def get_or_create(self, session_id: str) -> T:
        with self._lock:
            now = self._clock()
            self._evict_expired(now)

            entry = self._entries.get(session_id)
            if entry is not None:
                entry.last_used = now
                return entry.value

            if len(self._entries) >= self._max_sessions:
                oldest_id = min(self._entries, key=lambda sid: self._entries[sid].last_used)
                del self._entries[oldest_id]

            value = self._factory()
            self._entries[session_id] = _Entry(value=value, last_used=now)
            return value

    def __len__(self) -> int:
        return len(self._entries)

