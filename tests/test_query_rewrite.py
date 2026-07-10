import config
import rag


def _store():
    return {
        "dim": 3,
        "entries": [
            {"source": "a.md", "text": "exact match", "embedding": [1.0, 0.0, 0.0]},
        ],
    }


def test_rewrite_query_returns_rewritten_text(fake_client):
    client = fake_client([1.0, 0.0, 0.0], rewrite_reply="tom aspinall last fight")
    turns = [{"q": "who is champ?", "a": "Tom Aspinall"}]

    result = rag.rewrite_query(client, "when did he last fight?", turns)

    assert result == "tom aspinall last fight"
    assert len(client.models.generate_content_calls) == 1


def test_rewrite_query_falls_back_to_original_on_error(fake_client):
    client = fake_client([1.0, 0.0, 0.0], rewrite_reply=None)  # simulates generate_content raising

    result = rag.rewrite_query(client, "when did he last fight?", [{"q": "q", "a": "a"}])

    assert result == "when did he last fight?"


def test_rewrite_only_fires_on_follow_ups(fake_client, monkeypatch):
    client = fake_client([1.0, 0.0, 0.0], reply="ok", rewrite_reply="rewritten query")
    received_queries: list[str] = []

    def spy_retrieve(store, client_, question, **kwargs):
        received_queries.append(question)
        return [{"source": "a.md", "text": "x", "score": 0.9}]

    monkeypatch.setattr(rag, "retrieve", spy_retrieve)
    chat = rag.GroundedChat(_store(), client)

    chat.ask("who is champ?")
    chat.ask("when did he last fight?")

    assert received_queries[0] == "who is champ?"  # no rewrite on the first turn
    assert received_queries[1] == "rewritten query"  # rewrite fired on the follow-up
    assert len(client.models.generate_content_calls) == 1


def test_no_match_turn_not_recorded(fake_client, monkeypatch):
    client = fake_client([1.0, 0.0, 0.0])
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: [])
    chat = rag.GroundedChat(_store(), client)

    chat.ask("anything")

    assert chat._turns == []


def test_ask_stream_records_turn_only_after_full_consumption(fake_client):
    client = fake_client([1.0, 0.0, 0.0], reply="a b c")
    chat = rag.GroundedChat(_store(), client)

    _hits, chunks = chat.ask_stream("who is champ?")
    assert chat._turns == []  # generator not consumed yet

    list(chunks)

    assert chat._turns == [{"q": "who is champ?", "a": "abc"}]


def test_ask_stream_abandoned_generator_records_nothing(fake_client):
    client = fake_client([1.0, 0.0, 0.0], reply="a b c")
    chat = rag.GroundedChat(_store(), client)

    _hits, chunks = chat.ask_stream("who is champ?")
    next(chunks)  # partially consume, never exhaust

    assert chat._turns == []


def test_rewrite_disabled_via_config(fake_client, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_QUERY_REWRITE", False)
    client = fake_client([1.0, 0.0, 0.0], reply="ok", rewrite_reply="rewritten query")
    chat = rag.GroundedChat(_store(), client)

    chat.ask("who is champ?")
    chat.ask("when did he last fight?")

    assert client.models.generate_content_calls == []
