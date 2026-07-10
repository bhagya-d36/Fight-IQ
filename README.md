# RAG Chat — Python

A small, standalone Retrieval-Augmented Generation (RAG) chatbot prototype.
Put markdown files in `knowledge-base/`, embed them, and chat with an
assistant that answers only from that content.

## Setup (one time)

```
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell/cmd)
pip install -r requirements.txt
```

Then copy `.env.example` to `.env` and put your Gemini API key in it
(free key: https://aistudio.google.com/apikey). `.env.example` also lists
optional overrides (model names, `TOP_K`, `MIN_SIMILARITY`, timeouts, etc.) —
defaults live in `config.py`.

For development, install test/lint tooling too: `pip install -r requirements-dev.txt`.

## Usage

```
python ingest.py            # embed the knowledge base -> vector-store.json
python chat.py               # start chatting (terminal)
python ingest.py --dry-run   # preview chunking without API calls
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
on any host that has `GEMINI_API_KEY` set and `vector-store.json` present.

## Files

| File | Role |
|---|---|
| `ingest.py` | Chunk + embed the knowledge base into `vector-store.json` (incremental, `--force` to rebuild) |
| `rag.py` | Shared retrieval + grounded multi-turn chat logic (`GroundedChat`) used by both entry points |
| `sessions.py` | Bounded, TTL-evicting in-memory store used to keep one `GroundedChat` per web session |
| `config.py` | All tunable settings, overridable via env vars (see `.env.example`) |
| `chat.py` | Terminal chat: retrieve top chunks, grounded Gemini reply, multi-turn memory |
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
