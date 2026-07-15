"""embeddings.py — local sentence-embedding model shared by ingest.py and
rag.py. No API key or network call required after the model's first download.
"""

import config

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = _get_model().encode(texts, normalize_embeddings=True)
    return [[float(x) for x in v] for v in vectors]


def dimension() -> int:
    return _get_model().get_embedding_dimension()
