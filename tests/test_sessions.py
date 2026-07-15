from sessions import SessionStore


def _counting_factory():
    counter = {"n": 0}

    def factory():
        counter["n"] += 1
        return counter["n"]

    return factory, counter


def test_same_id_reuses_value():
    factory, counter = _counting_factory()
    store = SessionStore(factory=factory, max_sessions=10, ttl_seconds=100)

    v1 = store.get_or_create("a")
    v2 = store.get_or_create("a")

    assert v1 == v2
    assert counter["n"] == 1


def test_different_ids_get_different_values():
    factory, counter = _counting_factory()
    store = SessionStore(factory=factory, max_sessions=10, ttl_seconds=100)

    v1 = store.get_or_create("a")
    v2 = store.get_or_create("b")

    assert v1 != v2
    assert len(store) == 2


def test_lru_eviction_evicts_least_recently_used():
    factory, counter = _counting_factory()
    now = {"t": 0.0}
    store = SessionStore(factory=factory, max_sessions=2, ttl_seconds=1000, clock=lambda: now["t"])

    now["t"] = 1
    store.get_or_create("a")  # a created
    now["t"] = 2
    store.get_or_create("b")  # b created -> at capacity {a, b}
    now["t"] = 3
    store.get_or_create("a")  # touch a -> refreshes its last_used
    now["t"] = 4
    store.get_or_create("c")  # over capacity -> evicts LRU (b) to make room

    assert counter["n"] == 3
    assert len(store) == 2

    now["t"] = 5
    before = counter["n"]
    store.get_or_create("a")  # still cached -> no new factory call
    assert counter["n"] == before


def test_ttl_expiry_creates_fresh_value():
    factory, counter = _counting_factory()
    now = {"t": 0.0}
    store = SessionStore(factory=factory, max_sessions=10, ttl_seconds=60, clock=lambda: now["t"])

    v1 = store.get_or_create("a")
    now["t"] = 61  # idle past ttl_seconds
    v2 = store.get_or_create("a")

    assert v1 != v2
    assert counter["n"] == 2

