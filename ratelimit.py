"""ratelimit.py — fixed-window per-key rate limiter for the web UI.

Tracks a request count per key (client IP) within a sliding window of fixed
size; once a window fully elapses for a key, its counter resets. Stale keys
are pruned on access so the tracked set stays bounded to recently-active
clients.
"""

import threading
import time
from collections.abc import Callable


class RateLimiter:
    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._clock = clock
        self._windows: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    def _prune_expired(self, now: float) -> None:
        expired = [
            key
            for key, (window_start, _count) in self._windows.items()
            if now - window_start >= self._window_seconds
        ]
        for key in expired:
            del self._windows[key]

    def allow(self, key: str) -> bool:
        with self._lock:
            now = self._clock()
            self._prune_expired(now)

            window_start, count = self._windows.get(key, (now, 0))
            if count >= self._max_requests:
                self._windows[key] = (window_start, count)
                return False

            self._windows[key] = (window_start, count + 1)
            return True

    def __len__(self) -> int:
        return len(self._windows)
