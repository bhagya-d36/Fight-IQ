from rag import retrieve


def _store():
    return {
        "dim": 3,
        "entries": [
            {"source": "b.md", "text": "orthogonal", "embedding": [0.0, 1.0, 0.0]},
            {"source": "a.md", "text": "exact match", "embedding": [1.0, 0.0, 0.0]},
            {"source": "c.md", "text": "close match", "embedding": [0.9, 0.1, 0.0]},
        ],
    }


def test_top_k_ordering(fake_client):
    store = _store()
    client = fake_client([1.0, 0.0, 0.0])  # query embeds identical to a.md
    hits = retrieve(store, client, "question", top_k=3, min_similarity=-1.0)
    assert [h["source"] for h in hits] == ["a.md", "c.md", "b.md"]


def test_top_k_limits_results(fake_client):
    store = _store()
    client = fake_client([1.0, 0.0, 0.0])
    hits = retrieve(store, client, "question", top_k=1, min_similarity=-1.0)
    assert len(hits) == 1
    assert hits[0]["source"] == "a.md"


def test_min_similarity_filters_low_scores(fake_client):
    store = _store()
    client = fake_client([1.0, 0.0, 0.0])
    hits = retrieve(store, client, "question", top_k=3, min_similarity=0.5)
    sources = [h["source"] for h in hits]
    assert "b.md" not in sources  # orthogonal -> score 0.0, filtered out
    assert "a.md" in sources


def test_score_is_attached(fake_client):
    store = _store()
    client = fake_client([1.0, 0.0, 0.0])
    hits = retrieve(store, client, "question", top_k=3, min_similarity=-1.0)
    assert all("score" in h for h in hits)
    assert hits[0]["score"] > hits[-1]["score"]
