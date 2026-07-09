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


class FakeClient:
    def __init__(self, vector: list[float]):
        self.models = FakeModels(vector)


@pytest.fixture
def fake_client():
    def _make(vector: list[float]) -> FakeClient:
        return FakeClient(vector)

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
