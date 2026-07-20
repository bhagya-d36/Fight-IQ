import chromadb
import pytest

import config
import rag


def test_missing_collection_raises_file_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "CHROMA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        rag.load_store()


def test_empty_entries_raises_value_error():
    with pytest.raises(ValueError, match="ingest.py --force"):
        rag.validate_store({"model": rag.EMBEDDING_MODEL, "dim": 3, "entries": []})


def test_model_mismatch_raises_value_error():
    store = {
        "model": "some-other-model",
        "dim": 3,
        "entries": [{"source": "a.md", "text": "x"}],
    }
    with pytest.raises(ValueError, match="ingest.py --force"):
        rag.validate_store(store)


def test_dim_mismatch_raises_value_error():
    store = {
        "model": rag.EMBEDDING_MODEL,
        "dim": 512,
        "entries": [{"source": "a.md", "text": "x"}],
    }
    with pytest.raises(ValueError, match="ingest.py --force"):
        rag.validate_store(store)


def test_valid_store_passes():
    store = {
        "model": rag.EMBEDDING_MODEL,
        "dim": 3,
        "entries": [{"source": "a.md", "text": "x"}],
    }
    rag.validate_store(store)  # no exception


def test_load_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "CHROMA_DIR", tmp_path)
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection(
        rag.COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine", "model": config.EMBEDDING_MODEL, "dim": 3, "version": 3},
    )
    collection.add(ids=["a.md::0"], embeddings=[[0.1, 0.2, 0.3]], documents=["x"], metadatas=[{"source": "a.md"}])

    store = rag.load_store()

    assert store["entries"] == [{"source": "a.md", "text": "x"}]
    assert store["model"] == config.EMBEDDING_MODEL
    assert store["dim"] == 3
    assert store["version"] == 3
