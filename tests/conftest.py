"""Shared fixtures for the offline test suite — no network calls."""

from types import SimpleNamespace

import pytest


class FakeEmbedding:
    def __init__(self, values: list[float]):
        self.values = values


class FakeModels:
    """Duck-typed stand-in for genai.Client().models — returns a canned vector
    from embed_content and a canned (or failing) rewrite from generate_content.
    """

    def __init__(self, vector: list[float], rewrite_reply: str | None = "rewritten standalone query"):
        self._vector = vector
        self.rewrite_reply = rewrite_reply
        self.generate_content_calls: list[str] = []

    def embed_content(self, model, contents, config):
        return SimpleNamespace(embeddings=[FakeEmbedding(self._vector) for _ in contents])

    def generate_content(self, model, contents, config=None):
        self.generate_content_calls.append(contents)
        if self.rewrite_reply is None:
            raise RuntimeError("simulated generate_content failure")
        return SimpleNamespace(text=self.rewrite_reply)


class FakeChat:
    """Duck-typed stand-in for a google.genai chats.Chat session."""

    def __init__(self, reply: str, history: list | None = None):
        self.messages: list[str] = []
        self._history = list(history) if history else []
        self.reply = reply

    def send_message(self, message):
        self.messages.append(message)
        self._history += [f"user:{message}", f"model:{self.reply}"]
        return SimpleNamespace(text=self.reply)

    def send_message_stream(self, message):
        self.messages.append(message)
        self._history += [f"user:{message}", f"model:{self.reply}"]
        return iter(SimpleNamespace(text=piece) for piece in self.reply.split())

    def get_history(self, curated: bool = False):
        return list(self._history)


class FakeChats:
    """Duck-typed stand-in for genai.Client().chats — records every session created."""

    def __init__(self, reply: str):
        self._reply = reply
        self.created: list[FakeChat] = []
        self.create_history_args: list[list | None] = []

    def create(self, model, config=None, history=None):
        chat = FakeChat(self._reply, history=history)
        self.created.append(chat)
        self.create_history_args.append(history)
        return chat


class FakeClient:
    def __init__(
        self,
        vector: list[float],
        reply: str = "canned answer",
        rewrite_reply: str | None = "rewritten standalone query",
    ):
        self.models = FakeModels(vector, rewrite_reply)
        self.chats = FakeChats(reply)


@pytest.fixture
def fake_client():
    def _make(
        vector: list[float],
        reply: str = "canned answer",
        rewrite_reply: str | None = "rewritten standalone query",
    ) -> FakeClient:
        return FakeClient(vector, reply, rewrite_reply)

    return _make


@pytest.fixture
def sample_store():
    return {
        "version": 2,
        "model": "gemini-embedding-001",
        "dim": 3,
        "chunkChars": 1500,
        "files": {"a.md": "hash-a"},
        "entries": [
            {"source": "a.md", "text": "chunk one", "embedding": [1.0, 0.0, 0.0]},
            {"source": "a.md", "text": "chunk two", "embedding": [0.0, 1.0, 0.0]},
            {"source": "a.md", "text": "chunk three", "embedding": [0.9, 0.1, 0.0]},
        ],
    }
