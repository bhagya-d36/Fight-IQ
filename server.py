"""server.py — web frontend for the UFC/MMA RAG assistant.

Run `python ingest.py` first, then:  uvicorn server:app --reload
Open http://127.0.0.1:8000
"""

import json
import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from google.genai import errors
from pydantic import BaseModel

import rag  # importing rag loads config, which loads .env

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


class AskRequest(BaseModel):
    question: str


def _sources_payload(hits: list[dict]) -> list[dict]:
    return [{"source": h["source"], "score": round(h["score"], 2)} for h in hits]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        return {"answer": "", "grounded": False, "sources": []}

    try:
        hits = rag.retrieve(store, client, question)
        if not hits:
            return {
                "answer": "I don't have information about that in my knowledge base.",
                "grounded": False,
                "sources": [],
            }
        text = rag.answer(client, question, hits)
        return {"answer": text, "grounded": True, "sources": _sources_payload(hits)}
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


@app.get("/api/ask/stream")
def ask_stream(q: str) -> StreamingResponse:
    question = q.strip()

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def generate():
        if not question:
            yield sse("done", {})
            return

        try:
            hits = rag.retrieve(store, client, question)
            if not hits:
                yield sse("sources", {"grounded": False, "sources": []})
                yield sse(
                    "token",
                    {"text": "I don't have information about that in my knowledge base."},
                )
                return

            yield sse("sources", {"grounded": True, "sources": _sources_payload(hits)})
            for piece in rag.answer_stream(client, question, hits):
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
