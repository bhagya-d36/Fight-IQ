from ratelimit import RateLimiter


def test_allows_up_to_max_requests_per_window():
    now = {"t": 0.0}
    limiter = RateLimiter(max_requests=3, window_seconds=60, clock=lambda: now["t"])

    assert limiter.allow("a") is True
    assert limiter.allow("a") is True
    assert limiter.allow("a") is True


def test_denies_beyond_max_requests_per_window():
    now = {"t": 0.0}
    limiter = RateLimiter(max_requests=2, window_seconds=60, clock=lambda: now["t"])

    assert limiter.allow("a") is True
    assert limiter.allow("a") is True
    assert limiter.allow("a") is False


def test_resets_after_window_elapses():
    now = {"t": 0.0}
    limiter = RateLimiter(max_requests=1, window_seconds=60, clock=lambda: now["t"])

    assert limiter.allow("a") is True
    assert limiter.allow("a") is False

    now["t"] = 61  # window fully elapsed
    assert limiter.allow("a") is True


def test_distinct_keys_are_independent():
    now = {"t": 0.0}
    limiter = RateLimiter(max_requests=1, window_seconds=60, clock=lambda: now["t"])

    assert limiter.allow("a") is True
    assert limiter.allow("a") is False
    assert limiter.allow("b") is True


def test_stale_keys_are_pruned():
    now = {"t": 0.0}
    limiter = RateLimiter(max_requests=1, window_seconds=60, clock=lambda: now["t"])

    limiter.allow("a")
    limiter.allow("b")
    assert len(limiter) == 2

    now["t"] = 61  # both windows elapsed
    limiter.allow("c")  # triggers pruning
    assert len(limiter) == 1  # only "c" remains

