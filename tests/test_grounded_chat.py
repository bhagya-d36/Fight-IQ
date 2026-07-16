import rag


def _store():
    return {
        "dim": 3,
        "entries": [
            {"source": "a.md", "text": "exact match", "embedding": [1.0, 0.0, 0.0]},
        ],
    }


def test_system_instruction_includes_citation_guidance():
    assert "bracket number" in rag.SYSTEM_INSTRUCTION


def test_build_prompt_shape():
    hits = [{"source": "a.md", "text": "chunk text", "score": 0.9}]
    prompt = rag.build_prompt("who is champ?", hits)
    assert prompt.startswith("CONTEXT:\n")
    assert "(source: a.md)" in prompt
    assert "chunk text" in prompt
    assert prompt.endswith("QUESTION: who is champ?")


def test_ask_sends_context_and_question(fake_provider):
    provider = fake_provider(reply="the champ is X")
    chat = rag.GroundedChat(_store(), provider)

    text, hits = chat.ask("who is champ?")

    assert text == "the champ is X"
    assert hits
    sent = provider.chat_calls[0][-1]["content"]
    assert "CONTEXT:" in sent
    assert "QUESTION: who is champ?" in sent


def test_ask_no_hits_returns_no_match_without_sending(fake_provider, monkeypatch):
    provider = fake_provider()
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: [])
    chat = rag.GroundedChat(_store(), provider)

    text, hits = chat.ask("anything")

    assert text == rag.NO_MATCH_ANSWER
    assert hits == []
    assert provider.chat_calls == []


def test_ask_empty_model_reply(fake_provider):
    provider = fake_provider(reply="")
    chat = rag.GroundedChat(_store(), provider)

    text, _hits = chat.ask("who is champ?")

    assert text == rag.EMPTY_RESPONSE_ANSWER


def test_ask_stream_yields_reply_pieces(fake_provider):
    provider = fake_provider(reply="a b c")
    chat = rag.GroundedChat(_store(), provider)

    hits, chunks = chat.ask_stream("who is champ?")

    assert hits
    assert "".join(chunks) == "abc"


def test_ask_stream_no_hits_yields_no_match(fake_provider, monkeypatch):
    provider = fake_provider()
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: [])
    chat = rag.GroundedChat(_store(), provider)

    hits, chunks = chat.ask_stream("anything")

    assert hits == []
    assert list(chunks) == [rag.NO_MATCH_ANSWER]


def test_history_trimmed_after_max_turns(fake_provider):
    provider = fake_provider(reply="ok")
    chat = rag.GroundedChat(_store(), provider, max_turns=1)

    chat.ask("q1")
    chat.ask("q2")
    chat.ask("q3")

    assert len(chat._turns) == 1
    assert chat._turns[0]["q"] == "q3"


