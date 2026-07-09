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
(free key: https://aistudio.google.com/apikey).

## Usage

```
python ingest.py            # embed the knowledge base -> vector-store.json
python chat.py               # start chatting (terminal)
python ingest.py --dry-run   # preview chunking without API calls
```

Re-run `python ingest.py` every time you edit anything in `knowledge-base/`.

## Web UI

```
uvicorn server:app --reload
```

Then open `http://127.0.0.1:8000`. Same retrieval + grounding as `chat.py`, served
as a small FastAPI app with a hand-built frontend (no Node, no build step) in `web/`.

Override the bind address/port with env vars if needed: `HOST`, `PORT` (defaults
`127.0.0.1:8000`). To deploy, run `uvicorn server:app --host 0.0.0.0 --port $PORT`
on any host that has `GEMINI_API_KEY` set and `vector-store.json` present.

## Files

| File | Role |
|---|---|
| `ingest.py` | Chunk + embed the knowledge base into `vector-store.json` |
| `rag.py` | Shared retrieval + grounded-answer logic used by both entry points |
| `chat.py` | Terminal chat: retrieve top chunks, grounded Gemini reply |
| `server.py` | FastAPI app: `/api/ask`, `/api/ask/stream` (SSE), serves `web/` |
| `web/` | Hand-built frontend (`index.html`, `styles.css`, `app.js`) |
| `requirements.txt` | Dependencies (`google-genai`, `python-dotenv`, `fastapi`, `uvicorn`) |
| `knowledge-base/` | The knowledge base — add your own `.md` files here |

## Knowledge base structure

One `.md` file per topic in `knowledge-base/`. Each file has a single
`# Title` at the top, and one `## Section` per distinct fact — each `##`
section becomes one retrievable chunk. Keep sections self-contained and
under ~1,500 characters, and phrase FAQ headings as literal questions
people would actually ask.

See `tutorial/` for a full step-by-step walkthrough of how retrieval,
chunking, and grounding work in this project.
