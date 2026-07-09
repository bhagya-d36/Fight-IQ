# RAG Chat — Tutorial

A walkthrough of the standalone RAG chat prototype in this repo: what RAG is,
how this implementation works line by line, how to feed it knowledge, and how
it scales toward a production system.

Written for readers comfortable running Python, new to RAG.

## Chapters

| # | File | What you'll learn |
|---|---|---|
| 1 | [01-what-is-rag.md](01-what-is-rag.md) | The concept: embeddings, retrieval, grounding — and why RAG beats fine-tuning here |
| 2 | [02-setup-and-run.md](02-setup-and-run.md) | Get an API key, install, ingest, chat |
| 3 | [03-how-ingestion-works.md](03-how-ingestion-works.md) | What `ingest.py` does: chunking, embedding, the vector store |
| 4 | [04-how-chat-works.md](04-how-chat-works.md) | What `chat.py` does: retrieval, the grounded prompt, guardrails |
| 5 | [05-writing-kb-files.md](05-writing-kb-files.md) | How to author knowledge-base files that retrieve well |
| 6 | [06-tuning-and-troubleshooting.md](06-tuning-and-troubleshooting.md) | The knobs (TOP_K, threshold, chunk size — now in `config.py`/`.env`) and common failures |
| 7 | [07-path-to-production.md](07-path-to-production.md) | Mapping this prototype onto a production architecture (real database, real app) |

## The system in one paragraph

You put markdown files in `knowledge-base/`. `ingest.py` splits them into small
self-contained chunks and asks Gemini to turn each chunk into an *embedding* — a
list of 768 numbers that captures its meaning — then saves everything to
`vector-store.json`. When you ask a question in `chat.py`, your question gets
embedded the same way, the code finds the 4 chunks whose numbers are most
similar to the question's, and sends *only those chunks* to Gemini Flash with
strict instructions: answer from this context or say you don't know. That's
Retrieval-Augmented Generation — the model never answers from its own memory,
only from what you wrote.
