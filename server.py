"""server.py — web frontend for the UFC/MMA RAG assistant.

Run `python ingest.py` first, then:  uvicorn server:app --reload
Open http://127.0.0.1:8000
"""

import json
import logging
import os
import sys
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from google.genai import errors
from pydantic import BaseModel, Field

import config
import rag  # importing rag loads config, which loads .env
from ratelimit import RateLimiter
from sessions import SessionStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

FRIENDLY_ERROR = "Sorry — something went wrong talking to the model. Please try again."

try:
    store = rag.load_store()
except (FileNotFoundError, ValueError) as err:
    sys.exit(str(err))

try:
    client = rag.make_client()
except RuntimeError as err:
    sys.exit(str(err))

app = FastAPI(title="UFC RAG Assistant")

sessions: SessionStore[rag.GroundedChat] = SessionStore(
    factory=lambda: rag.GroundedChat(store, client),
    max_sessions=config.MAX_SESSIONS,
    ttl_seconds=config.SESSION_TTL_MINUTES * 60,
)

limiter = RateLimiter(config.RATE_LIMIT_REQUESTS, config.RATE_LIMIT_WINDOW_SECONDS)

MAX_SESSION_ID_LEN = 64


class AskRequest(BaseModel):
    question: str = Field(max_length=config.MAX_QUESTION_CHARS)
    session_id: str | None = None


def _chat_for(session_id: str | None) -> rag.GroundedChat:
    """Reuse a session's chat if a valid id is given, else a one-shot chat."""
    if session_id and len(session_id) <= MAX_SESSION_ID_LEN:
        return sessions.get_or_create(session_id)
    return rag.GroundedChat(store, client)


def rate_limit(request: Request) -> None:
    if not config.RATE_LIMIT_ENABLED:
        return
    client_host = request.client.host if request.client else "unknown"
    if not limiter.allow(client_host):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please slow down.")


def _sources_payload(hits: list[dict]) -> list[dict]:
    return [{"source": h["source"], "score": round(h["score"], 2)} for h in hits]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "chunks": len(store["entries"]),
        "model": store.get("model"),
        "storeVersion": store.get("version"),
        "sessions": len(sessions),
    }


@app.post("/api/ask", dependencies=[Depends(rate_limit)])
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        return {"answer": "", "grounded": False, "sources": []}

    try:
        chat = _chat_for(req.session_id)
        text, hits = chat.ask(question)
        return {"answer": text, "grounded": bool(hits), "sources": _sources_payload(hits)}
    except errors.APIError as err:
        logger.exception("Gemini API error answering question")
        return JSONResponse(
            status_code=502,
            content={"answer": FRIENDLY_ERROR, "grounded": False, "sources": [], "error": str(err)},
        )
    except Exception as err:
        logger.exception("Unexpected error answering question")
        return JSONResponse(
            status_code=500,
            content={"answer": FRIENDLY_ERROR, "grounded": False, "sources": [], "error": str(err)},
        )


@app.get("/api/ask/stream", dependencies=[Depends(rate_limit)])
def ask_stream(
    q: str = Query(max_length=config.MAX_QUESTION_CHARS),
    session_id: str | None = None,
) -> StreamingResponse:
    question = q.strip()

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def generate():
        if not question:
            yield sse("done", {})
            return

        try:
            chat = _chat_for(session_id)
            hits, chunks = chat.ask_stream(question)
            yield sse("sources", {"grounded": bool(hits), "sources": _sources_payload(hits)})
            for piece in chunks:
                yield sse("token", {"text": piece})
        except Exception:
            logger.exception("Error while streaming an answer")
            yield sse("token", {"text": FRIENDLY_ERROR})
        finally:
            yield sse("done", {})

    return StreamingResponse(generate(), media_type="text/event-stream")


app.mount("/", StaticFiles(directory=WEB_DIR), name="web")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=True,
    )
