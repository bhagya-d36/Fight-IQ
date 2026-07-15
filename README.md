# FightIQ

An AI-powered knowledge library for Mixed Martial Arts (Primarily UFC), with context grounded using Retrieval-Augmented Generation (RAG).
Contains informations regarding champions, rankings, rules, weight classes, and event history. Put markdown
files in `knowledge-base/`, embed them, and chat with an assistant that
answers only from that content, with citations back to the source.

## Setup (one time)

```
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell/cmd)
pip install -r requirements.txt
```

Then copy `.env.example` to `.env`, set `LLM_PROVIDER` to whichever chat LLM
you want (`gemini`, `openai`, `anthropic`, `deepseek`, or `kimi`), and put that
provider's API key in it. `.env.example` also lists optional overrides (model
names, `TOP_K`, `MIN_SIMILARITY`, timeouts, etc.) — defaults live in
`config.py`. Embeddings run locally (`sentence-transformers`, no key needed),
so `ingest.py` never requires an API key.

For development, install test/lint tooling too: `pip install -r requirements-dev.txt`.

## Usage

```
python ingest.py            # embed the knowledge base -> vector-store.json
python chat.py               # start chatting (terminal)
python ingest.py --dry-run   # preview chunking without embedding
python ingest.py --force     # re-embed every file, ignoring the cache
```

Re-run `python ingest.py` every time you edit anything in `knowledge-base/` —
it hashes each file and only re-embeds ones that actually changed, so this is
cheap even with a large knowledge base.

## Web UI

```
uvicorn server:app --reload
```

Then open `http://127.0.0.1:8000`. Same retrieval + grounding as `chat.py`, served
as a small FastAPI app with a hand-built frontend (no Node, no build step) in `web/`.

The browser keeps a client-generated session id (`localStorage`) and sends it
with every question, so follow-ups like "and his last fight?" resolve like they
do in `chat.py`. Sessions live in memory on the server only — bounded by
`MAX_SESSIONS` (LRU-evicted past that) and `SESSION_TTL_MINUTES` (idle
sessions expire), with each session's own chat history capped at
`MAX_CHAT_TURNS`. A server restart drops all sessions.

Override the bind address/port with env vars if needed: `HOST`, `PORT` (defaults
`127.0.0.1:8000`). To deploy, run `uvicorn server:app --host 0.0.0.0 --port $PORT`
on any host that has the chosen provider's API key set and `vector-store.json` present.

The server also exposes `GET /health` (chunk count, model, store version, live
session count) for deploy probes, caps question length at `MAX_QUESTION_CHARS`
(422 if exceeded), and rate-limits `/api/ask*` per client IP — `RATE_LIMIT_REQUESTS`
per `RATE_LIMIT_WINDOW_SECONDS`, `RATE_LIMIT_ENABLED` to turn it off. The limiter
is in-memory per process (a multi-worker deploy counts each worker separately)
and keyed by the direct peer IP (not `X-Forwarded-For`, which is spoofable).

## Answer quality

A few things beyond plain vector search improve retrieval and answers:

- **Query rewriting** — on follow-up turns, the raw question is rewritten into a
  standalone search query using recent conversation context before retrieval
  (so "when did he last fight?" resolves to the fighter named in the prior
  answer). Toggle with `ENABLE_QUERY_REWRITE`; window size via `REWRITE_HISTORY_TURNS`.
- **Chunk overlap** — packed chunks in oversized sections carry a small tail of
  the previous chunk forward (`MAX_CHUNK_OVERLAP`), so a fact split across a
  packing boundary isn't orphaned. Changing it requires `python ingest.py --force`
  to re-embed.
- **Hybrid search** — a BM25 keyword pass is fused with vector search via
  reciprocal rank fusion, so exact names/records aren't lost to semantically-similar
  chunks. The no-match decision stays cosine-only — keyword overlap alone can't
  manufacture an answer. Toggle with `HYBRID_SEARCH`; fusion constant via `RRF_K`.
- **Inline citations** — answers cite `[1][2]`-style bracket numbers back to the
  numbered CONTEXT excerpts; the web UI links them to the matching source row.

## Files

| File | Role |
|---|---|
| `ingest.py` | Chunk + locally embed the knowledge base into `vector-store.json` (incremental, `--force` to rebuild) |
| `embeddings.py` | Local sentence-transformers embedding model (no API key) |
| `llm.py` | Chat-provider abstraction (`ChatProvider`) + adapters for Gemini, OpenAI-compatible, and Anthropic |
| `rag.py` | Shared retrieval + grounded multi-turn chat logic (`GroundedChat`) used by both entry points |
| `sessions.py` | Bounded, TTL-evicting in-memory store used to keep one `GroundedChat` per web session |
| `ratelimit.py` | Fixed-window per-IP rate limiter used by `server.py` |
| `config.py` | All tunable settings, overridable via env vars (see `.env.example`) |
| `chat.py` | Terminal chat: retrieve top chunks, grounded LLM reply, multi-turn memory |
| `server.py` | FastAPI app: `/api/ask`, `/api/ask/stream` (SSE), session-aware, serves `web/` |
| `web/` | Hand-built frontend (`index.html`, `styles.css`, `app.js`) |
| `tests/` | Offline pytest suite (`pytest -q`) — no network calls |
| `requirements.txt` | Pinned runtime dependencies |
| `requirements-dev.txt` | Runtime deps plus `pytest`/`ruff` for development |
| `knowledge-base/` | The knowledge base — add your own `.md` files here |

## Development

```
pip install -r requirements-dev.txt
pytest -q          # run the test suite (offline, no API key needed)
ruff check .       # lint
```

If this folder lives inside a cloud-synced directory (OneDrive, Dropbox), be
aware the sync client can occasionally lock `.git` files mid-operation —
retry the git command if that happens.

## Knowledge base structure

One `.md` file per topic in `knowledge-base/`. Each file has a single
`# Title` at the top, and one `## Section` per distinct fact — each `##`
section becomes one retrievable chunk. Keep sections self-contained and
under ~1,500 characters, and phrase FAQ headings as literal questions
people would actually ask.

See `tutorial/` for a full step-by-step walkthrough of how retrieval,
chunking, and grounding work in this project.
