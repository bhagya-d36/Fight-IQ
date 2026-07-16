# 3. How ingestion works (`ingest.py`)

The ingest pipeline has three stages: **read → chunk → embed → store**. It runs
offline, before any chat happens, and its output is `vector-store.json`.

## Stage 1: Read

Every `.md` file in `knowledge-base/` is read. Nothing else — the folder is the
single attachment point for knowledge.

## Stage 2: Chunk — the step that decides retrieval quality

You can't embed a whole file as one vector: a file mixing pricing, support
policy, and setup instructions would average into a vector that matches
everything weakly and nothing well. You also don't want chunks so tiny they
lose context. This project chunks by **markdown structure**:

1. Split each file on `## ` headings — each section becomes one chunk.
2. If a section exceeds **1,500 characters** (`MAX_CHUNK_CHARS`), it is further
   split on paragraph breaks, packing paragraphs greedily up to the limit.
3. Sections that are only a title with no body are skipped.

### The prefix trick

Every chunk gets prefixed with `Document Title — Section Heading`:

```
Example Topic — How should I format my files?

## How should I format my files?
Start each file with a single # Title, then add one ## Section...
```

Why: after retrieval, the LLM sees the chunk **in isolation** — it has no idea
which file it came from. The prefix keeps each chunk self-describing, and it
also improves matching, because words from the title are now *inside* the
embedded text.

This is why the KB writing rules (chapter 5) say "one `##` per fact": **your
heading structure literally is your chunking strategy.**

## Stage 3: Embed

Chunks are sent to `gemini-embedding-001` in batches of 50:

```python
res = client.models.embed_content(
    model="gemini-embedding-001",
    contents=texts,
    config=types.EmbedContentConfig(
        output_dimensionality=768,
        task_type="RETRIEVAL_DOCUMENT",
    ),
)
```

Two details worth understanding:

- **`output_dimensionality=768`** — the model can emit larger vectors, but 768
  is plenty for a small KB, keeps the JSON file small, and matches the
  standard `vector(768)` column size used by pgvector, so nothing changes if
  you migrate to a real database later.
- **`task_type="RETRIEVAL_DOCUMENT"`** — Gemini embeddings are *asymmetric*:
  stored documents are embedded as `RETRIEVAL_DOCUMENT`, questions (in
  `chat.py`) as `RETRIEVAL_QUERY`. The model shapes the vectors so short
  questions land near the longer documents that answer them. Mixing these up
  silently degrades match quality.

## Stage 4: Store

Everything is written to `vector-store.json`:

```json
{
  "version": 2,
  "model": "gemini-embedding-001",
  "dim": 768,
  "chunkChars": 1500,
  "createdAt": "...",
  "files": { "example.md": "<sha256 of the file's content>" },
  "entries": [
    { "source": "example.md", "text": "Example Topic — ...", "embedding": [0.01, ...] }
  ]
}
```

A JSON file *is* a legitimate vector store at this scale: a few dozen chunks
× 768 floats is nothing. A real database (pgvector) becomes worth it when you
have thousands of chunks, concurrent users, or need to update single
documents without re-reading everything — see chapter 7.

## Incremental re-ingestion

`ingest.py` hashes each KB file's content and stores the hashes under `files`.
On the next run, a file whose hash hasn't changed has its existing chunks and
embeddings **reused** instead of re-embedded — only new or edited files cost
an API call. This makes `python ingest.py` cheap to re-run habitually.

| You changed... | Re-run ingest? | Cost |
|---|---|---|
| A file's content in `knowledge-base/` | **Yes** | Only that file re-embeds |
| `TOP_K`, `MIN_SIMILARITY` (in `config.py`/`.env`) | No — those are query-time knobs | — |
| `MAX_CHUNK_CHARS` or the chunking code | **Yes** | Full rebuild (chunk boundaries changed) |
| The embedding model or dimensions | **Yes** | Full rebuild — old and new vectors are incompatible |

`python ingest.py --force` always rebuilds everything from scratch, ignoring
the cache — use it after a chunking-logic or model change, or if you suspect
the store is out of sync.

---

Next: [4. How chat works](04-how-chat-works.md)


