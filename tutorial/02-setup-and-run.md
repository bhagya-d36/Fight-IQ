# 2. Setup and run

## Prerequisites

- Python 3.10+ (`python --version`).
- A Google account for the API key.

## Step 1 — Get a Gemini API key

1. Go to https://aistudio.google.com/apikey
2. Click **Create API key** (free tier is fine for this prototype).
3. In this folder, copy `.env.example` to `.env` and paste the key:

```
GEMINI_API_KEY=AIza...your-actual-key
```

`.env` is gitignored — the key never leaves your machine.

## Step 2 — Install dependencies

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Installs exactly two packages: `google-genai` (the official Gemini SDK) and
`python-dotenv` (loads `.env`).

## Step 3 — Ingest the knowledge base

```
python ingest.py
```

Expected output (with the example file that ships in `knowledge-base/`):

```
example.md: 4 chunk(s)
Embedding 4 chunks with gemini-embedding-001...
  4/4
Done. Wrote 4 chunks to vector-store.json
```

This creates `vector-store.json` — the entire "vector database" of this
prototype. **Re-run this command every time you edit any file in
`knowledge-base/`**, or the chat will keep answering from the old content.

To preview chunking without spending any API calls:

```
python ingest.py --dry-run
```

## Step 4 — Chat

```
python chat.py
```

```
RAG chat — gemini-2.5-flash over 4 KB chunks.
Type a question, or "exit" to quit.

you > how do I format my knowledge base files?

bot > Start each file with a single # Title, then one ## section per
      distinct fact — each ## section becomes one retrievable chunk.

      [retrieved: example.md (0.68)]
```

The `[retrieved: ...]` line shows which knowledge-base files the answer came
from and the similarity score (0–1). Watch it while testing — it tells you
whether retrieval found the right material *before* you judge the answer.

## Good first experiments

- Ask something covered by the example KB file in different words.
- Ask something NOT in the KB: *"what's the weather today?"* — it should
  decline rather than invent an answer.
- Ask a follow-up question — the chat keeps history, so follow-ups work.
- Replace `knowledge-base/example.md` with your own content, re-run
  `python ingest.py`, and confirm the bot answers from your new material.

---

Next: [3. How ingestion works](03-how-ingestion-works.md)
