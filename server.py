"""server.py — web frontend for the UFC/MMA RAG assistant.

Run `python ingest.py` first, then:  uvicorn server:app --reload
Open http://127.0.0.1:8000
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import rag

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

try:
    store = rag.load_store()
except FileNotFoundError as err:
    sys.exit(str(err))

try:
    client = rag.make_client()
except RuntimeError as err:
    sys.exit(str(err))

app = FastAPI(title="UFC RAG Assistant")


class AskRequest(BaseModel):
    question: str


def _sources_payload(hits: list[dict]) -> list[dict]:
    return [{"source": h["source"], "score": round(h["score"], 2)} for h in hits]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    question = req.question.strip()
    if not question:
        return {"answer": "", "grounded": False, "sources": []}

    hits = rag.retrieve(store, client, question)
    if not hits:
        return {
            "answer": "I don't have information about that in my knowledge base.",
            "grounded": False,
            "sources": [],
        }

    text = rag.answer(client, question, hits)
    return {"answer": text, "grounded": True, "sources": _sources_payload(hits)}


@app.get("/api/ask/stream")
def ask_stream(q: str) -> StreamingResponse:
    question = q.strip()

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def generate():
        if not question:
            yield sse("done", {})
            return

        hits = rag.retrieve(store, client, question)
        if not hits:
            yield sse("sources", {"grounded": False, "sources": []})
            yield sse(
                "token",
                {"text": "I don't have information about that in my knowledge base."},
            )
            yield sse("done", {})
            return

        yield sse("sources", {"grounded": True, "sources": _sources_payload(hits)})
        for piece in rag.answer_stream(client, question, hits):
            yield sse("token", {"text": piece})
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
