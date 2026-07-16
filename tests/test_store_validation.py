import json

import pytest

import rag


def test_missing_file_raises_file_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "STORE_FILE", tmp_path / "does-not-exist.json")
    with pytest.raises(FileNotFoundError):
        rag.load_store()


def test_corrupt_json_raises_value_error(tmp_path, monkeypatch):
    store_file = tmp_path / "vector-store.json"
    store_file.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(rag, "STORE_FILE", store_file)
    with pytest.raises(ValueError, match="corrupt"):
        rag.load_store()


def test_empty_entries_raises_value_error():
    with pytest.raises(ValueError, match="ingest.py --force"):
        rag.validate_store({"model": rag.EMBEDDING_MODEL, "dim": 3, "entries": []})


def test_model_mismatch_raises_value_error():
    store = {
        "model": "some-other-model",
        "dim": 3,
        "entries": [{"source": "a.md", "text": "x", "embedding": [0.0] * 3}],
    }
    with pytest.raises(ValueError, match="ingest.py --force"):
        rag.validate_store(store)


def test_dim_mismatch_raises_value_error():
    store = {
        "model": rag.EMBEDDING_MODEL,
        "dim": 512,
        "entries": [{"source": "a.md", "text": "x", "embedding": [0.0] * 512}],
    }
    with pytest.raises(ValueError, match="ingest.py --force"):
        rag.validate_store(store)


def test_embedding_length_mismatch_raises_value_error():
    store = {
        "model": rag.EMBEDDING_MODEL,
        "dim": 3,
        "entries": [{"source": "a.md", "text": "x", "embedding": [0.0] * 100}],
    }
    with pytest.raises(ValueError, match="ingest.py --force"):
        rag.validate_store(store)


def test_valid_store_passes():
    store = {
        "model": rag.EMBEDDING_MODEL,
        "dim": 3,
        "entries": [{"source": "a.md", "text": "x", "embedding": [0.0] * 3}],
    }
    rag.validate_store(store)  # no exception


def test_load_store_roundtrip(tmp_path, monkeypatch):
    store = {
        "model": rag.EMBEDDING_MODEL,
        "dim": 3,
        "entries": [{"source": "a.md", "text": "x", "embedding": [0.1, 0.2, 0.3]}],
    }
    store_file = tmp_path / "vector-store.json"
    store_file.write_text(json.dumps(store), encoding="utf-8")
    monkeypatch.setattr(rag, "STORE_FILE", store_file)
    loaded = rag.load_store()
    assert loaded == store


