"""Shared fixtures for the offline test suite — no network calls."""

from types import SimpleNamespace

import pytest


class FakeEmbedding:
    def __init__(self, values: list[float]):
        self.values = values


class FakeModels:
    """Duck-typed stand-in for genai.Client().models — returns a canned vector."""

    def __init__(self, vector: list[float]):
        self._vector = vector

    def embed_content(self, model, contents, config):
        return SimpleNamespace(embeddings=[FakeEmbedding(self._vector) for _ in contents])


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
    def __init__(self, vector: list[float], reply: str = "canned answer"):
        self.models = FakeModels(vector)
        self.chats = FakeChats(reply)


@pytest.fixture
def fake_client():
    def _make(vector: list[float], reply: str = "canned answer") -> FakeClient:
        return FakeClient(vector, reply)

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
