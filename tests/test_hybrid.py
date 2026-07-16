import rag


def _keyword_store():
    return {
        "dim": 3,
        "entries": [
            {
                "source": "a.md",
                "text": "The event took place in Las Vegas Nevada last year.",
                "embedding": [1.0, 0.0, 0.0],  # highest cosine, no keyword overlap
            },
            {
                "source": "b.md",
                "text": "Tom Aspinall is the current UFC heavyweight champion.",
                "embedding": [0.6, 0.8, 0.0],  # lower cosine, strong keyword overlap
            },
        ],
    }


def test_rrf_lifts_keyword_match_above_higher_cosine_nonmatch():
    store = _keyword_store()

    hits = rag.retrieve(store, "aspinall heavyweight champion", top_k=2, hybrid=True)

    assert [h["source"] for h in hits] == ["b.md", "a.md"]


def test_hybrid_gate_blocks_low_cosine_even_with_keyword_match():
    store = {
        "dim": 3,
        "entries": [
            {"source": "a.md", "text": "Tom Aspinall is the heavyweight champion.", "embedding": [0.0, 1.0, 0.0]},
        ],
    }

    hits = rag.retrieve(store, "aspinall heavyweight champion", top_k=3, min_similarity=0.35, hybrid=True)

    assert hits == []


def test_hybrid_falls_back_to_pure_cosine_ordering_when_no_keyword_match():
    # Mirrors test_retrieve.py's exact-ordering scenario: the query shares no
    # tokens with any entry, so BM25 is all-zero and hybrid degrades to the
    # unchanged pure-vector path.
    store = {
        "dim": 3,
        "entries": [
            {"source": "b.md", "text": "orthogonal content here", "embedding": [0.0, 1.0, 0.0]},
            {"source": "a.md", "text": "exact match content here", "embedding": [1.0, 0.0, 0.0]},
            {"source": "c.md", "text": "close match content here", "embedding": [0.9, 0.1, 0.0]},
        ],
    }

    hits = rag.retrieve(store, "question", top_k=3, min_similarity=-1.0, hybrid=True)

    assert [h["source"] for h in hits] == ["a.md", "c.md", "b.md"]


def test_hybrid_false_forces_pure_vector():
    store = _keyword_store()

    hits = rag.retrieve(store, "aspinall heavyweight champion", top_k=2, hybrid=False)

    assert [h["source"] for h in hits] == ["a.md", "b.md"]


def test_displayed_score_is_cosine_even_when_order_is_fused():
    store = _keyword_store()

    hits = rag.retrieve(store, "aspinall heavyweight champion", top_k=2, hybrid=True)

    scores = {h["source"]: h["score"] for h in hits}
    assert scores["a.md"] == 1.0
    assert abs(scores["b.md"] - 0.6) < 1e-9


def test_tokenize_keeps_records_and_unicode_names():
    tokens = rag._tokenize("Record 12-3, fighter Procházka faced the champion")

    assert "12-3" in tokens
    assert "procházka" in tokens


