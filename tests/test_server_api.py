import sys

import pytest
from fastapi.testclient import TestClient

import config
import llm
import rag

SAMPLE_STORE = {
    "dim": 3,
    "entries": [
        {"source": "a.md", "text": "exact match", "embedding": [1.0, 0.0, 0.0]},
    ],
}


@pytest.fixture
def server_app(monkeypatch, fake_provider):
    provider = fake_provider(reply="canned answer")
    monkeypatch.setattr(rag, "load_store", lambda: SAMPLE_STORE)
    monkeypatch.setattr(llm, "make_chat_provider", lambda: provider)
    sys.modules.pop("server", None)
    import server  # re-imports fresh so it binds store/provider via the patched functions

    yield server, provider
    sys.modules.pop("server", None)


def test_ask_happy_path(server_app):
    server, _provider = server_app
    with TestClient(server.app) as tc:
        res = tc.post("/api/ask", json={"question": "who is champ?"})

    assert res.status_code == 200
    data = res.json()
    assert data["grounded"] is True
    assert data["answer"] == "canned answer"
    assert data["sources"][0]["source"] == "a.md"


def test_ask_empty_question_short_circuits(server_app):
    server, provider = server_app
    with TestClient(server.app) as tc:
        res = tc.post("/api/ask", json={"question": "   "})

    assert res.json() == {"answer": "", "grounded": False, "sources": []}
    assert provider.chat_calls == []


def test_ask_same_session_id_reuses_chat(server_app):
    server, provider = server_app
    with TestClient(server.app) as tc:
        tc.post("/api/ask", json={"question": "q1", "session_id": "s1"})
        tc.post("/api/ask", json={"question": "q2", "session_id": "s1"})

    assert len(provider.chat_calls) == 2
    # second call's history includes the first turn's Q&A plus the new prompt
    assert len(provider.chat_calls[1]) == 3


def test_ask_different_session_ids_get_different_chats(server_app):
    server, provider = server_app
    with TestClient(server.app) as tc:
        tc.post("/api/ask", json={"question": "q1", "session_id": "s1"})
        tc.post("/api/ask", json={"question": "q2", "session_id": "s2"})

    assert len(provider.chat_calls) == 2
    assert len(provider.chat_calls[0]) == 1
    assert len(provider.chat_calls[1]) == 1


def test_stream_events_in_order(server_app):
    server, _provider = server_app
    with TestClient(server.app) as tc, tc.stream(
        "GET", "/api/ask/stream", params={"q": "who is champ?", "session_id": "s1"}
    ) as res:
        body = "".join(res.iter_text())

    events = [line.split(": ", 1)[1] for line in body.splitlines() if line.startswith("event:")]
    assert events[0] == "sources"
    assert events[-1] == "done"
    assert "token" in events


def test_stream_empty_question_only_emits_done(server_app):
    server, _provider = server_app
    with TestClient(server.app) as tc, tc.stream("GET", "/api/ask/stream", params={"q": "  "}) as res:
        body = "".join(res.iter_text())

    events = [line.split(": ", 1)[1] for line in body.splitlines() if line.startswith("event:")]
    assert events == ["done"]


def test_health_endpoint(server_app):
    server, _provider = server_app
    with TestClient(server.app) as tc:
        res = tc.get("/health")

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["chunks"] == 1


def test_ask_rejects_question_over_max_length(server_app):
    server, _provider = server_app
    with TestClient(server.app) as tc:
        res = tc.post("/api/ask", json={"question": "x" * (config.MAX_QUESTION_CHARS + 1)})

    assert res.status_code == 422


def test_stream_rejects_query_over_max_length(server_app):
    server, _provider = server_app
    with TestClient(server.app) as tc:
        res = tc.get("/api/ask/stream", params={"q": "x" * (config.MAX_QUESTION_CHARS + 1)})

    assert res.status_code == 422


def test_rate_limit_returns_429_when_exceeded(monkeypatch, fake_provider):
    provider = fake_provider(reply="canned answer")
    monkeypatch.setattr(rag, "load_store", lambda: SAMPLE_STORE)
    monkeypatch.setattr(llm, "make_chat_provider", lambda: provider)
    monkeypatch.setattr(config, "RATE_LIMIT_REQUESTS", 1)
    sys.modules.pop("server", None)
    import server

    try:
        with TestClient(server.app) as tc:
            res1 = tc.post("/api/ask", json={"question": "q1"})
            res2 = tc.post("/api/ask", json={"question": "q2"})
    finally:
        sys.modules.pop("server", None)

    assert res1.status_code == 200
    assert res2.status_code == 429


def test_health_not_rate_limited(monkeypatch, fake_provider):
    provider = fake_provider(reply="canned answer")
    monkeypatch.setattr(rag, "load_store", lambda: SAMPLE_STORE)
    monkeypatch.setattr(llm, "make_chat_provider", lambda: provider)
    monkeypatch.setattr(config, "RATE_LIMIT_REQUESTS", 1)
    sys.modules.pop("server", None)
    import server

    try:
        with TestClient(server.app) as tc:
            tc.post("/api/ask", json={"question": "q1"})
            tc.post("/api/ask", json={"question": "q2"})  # exhausts the limit
            health_res = tc.get("/health")
    finally:
        sys.modules.pop("server", None)

    assert health_res.status_code == 200


