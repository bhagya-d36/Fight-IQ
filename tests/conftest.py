"""Shared fixtures for the offline test suite — no network calls."""

import pytest

import embeddings


class FakeChatProvider:
    """Duck-typed stand-in for an llm.ChatProvider — returns a canned reply
    from chat/stream_chat and a canned (or failing) rewrite from complete.
    """

    def __init__(self, reply: str = "canned answer", rewrite_reply: str | None = "rewritten standalone query"):
        self.reply = reply
        self.rewrite_reply = rewrite_reply
        self.chat_calls: list[list[dict]] = []
        self.complete_calls: list[str] = []

    def chat(self, messages, system=None, temperature=0.2):
        self.chat_calls.append(messages)
        return self.reply

    def stream_chat(self, messages, system=None, temperature=0.2):
        self.chat_calls.append(messages)
        return iter(self.reply.split())

    def complete(self, prompt, temperature=0.0):
        self.complete_calls.append(prompt)
        if self.rewrite_reply is None:
            raise RuntimeError("simulated complete failure")
        return self.rewrite_reply


@pytest.fixture
def fake_provider():
    def _make(reply: str = "canned answer", rewrite_reply: str | None = "rewritten standalone query"):
        return FakeChatProvider(reply, rewrite_reply)

    return _make


@pytest.fixture(autouse=True)
def stub_embeddings(monkeypatch):
    """Every test store in this suite is built around a query embedding of
    [1.0, 0.0, 0.0] against 3-dim entries. Stub the real (torch-backed) model
    so the suite stays fast, offline, and dependency-free.
    """
    monkeypatch.setattr(embeddings, "embed_texts", lambda texts: [[1.0, 0.0, 0.0] for _ in texts])
    monkeypatch.setattr(embeddings, "dimension", lambda: 3)


@pytest.fixture
def sample_store():
    return {
        "version": 2,
        "model": "all-MiniLM-L6-v2",
        "dim": 3,
        "chunkChars": 1500,
        "files": {"a.md": "hash-a"},
        "entries": [
            {"source": "a.md", "text": "chunk one", "embedding": [1.0, 0.0, 0.0]},
            {"source": "a.md", "text": "chunk two", "embedding": [0.0, 1.0, 0.0]},
            {"source": "a.md", "text": "chunk three", "embedding": [0.9, 0.1, 0.0]},
        ],
    }


