# 7. Path to production

This prototype is a minimal version of what a production RAG system looks
like with every non-essential part stripped away. The concepts transfer
one-to-one; only the containers change.

## Component mapping

| This prototype | Production | What changes |
|---|---|---|
| `knowledge-base/*.md` folder | A `documents` table (title, source_type, content, active) | Docs get CRUD in an admin panel instead of file edits |
| Chunks inside `vector-store.json` | A `chunks` table (document_id, chunk_text, `embedding vector(768)`, token_count) | Same 768-dim vectors — chosen here deliberately so they carry over |
| `python ingest.py` | A `reindex` admin endpoint | Triggered after a doc edit; re-embeds only the changed document — the prototype already does this (content-hash based, see chapter 3) |
| Brute-force cosine loop in `chat.py` | pgvector similarity query (`ORDER BY embedding <=> $1 LIMIT 4`) | Same math, done in Postgres with an index |
| Terminal input loop | Web/app/chat-widget frontend → backend → this retrieval logic | The retrieval logic is called from whatever channel your users are on |
| Fixed "I don't know" string | Escalation: open a support ticket for a human | The human-in-the-loop guarantee for low-confidence answers |
| `[retrieved: ...]` console line | Log every turn (question, retrieved chunk ids, answer) | Auditability — you can review what the LLM actually said |
| Facts written in markdown | Live facts fetched from your own database/API where correctness matters | See below — the biggest behavioral change |

## The two upgrades that matter most

### 1. Critical facts leave the generation path entirely

Here, the model quotes facts from retrieved markdown, and a system-prompt rule
keeps it honest. A production system goes further for anything that must never
be wrong (prices, account balances, inventory): retrieval identifies *which*
record the user means, then a reply template pulls the value straight from
your database. The model never gets the chance to alter it. Prompt rules are
a seatbelt; taking the model out of the critical-data path is the actual
guardrail.

### 2. Classification before answering

Here, every input goes to RAG. In production, free text often first hits an
intent classifier (also a small/fast LLM call) that picks one of three routes:

- **(a) resume a rule-based flow** — "actually I want to do X" → back to your
  application logic
- **(b) RAG answer** — an informational question → this prototype's flow
- **(c) escalate** — complaint, low confidence, or "I want a human" → ticket

The prototype's `MIN_SIMILARITY` floor is a primitive version of route (c).

## What stays exactly the same

- The asymmetric embedding pattern (`RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`).
- 768-dim `gemini-embedding-001` vectors.
- Heading-based chunking with the title prefix — the KB authoring rules in
  chapter 5 apply verbatim to content typed into a production documents table.
- The grounded system prompt and its four rules.
- The top-k + threshold retrieval shape.

## Migration sketch (when the time comes)

1. Create `documents` + `chunks` tables in your database, enable pgvector.
2. Port `chunk_markdown()` from `ingest.py` into your reindex endpoint as-is.
3. Insert the current `.md` files as the first `documents` rows.
4. Replace `retrieve()`'s sort/slice with one SQL query.
5. Wire this retrieval logic behind whatever channel/interface your users
   already talk to, and route its escalations to your support/ticketing system.

Nothing in steps 1–5 changes what you learned in chapters 1–6 — that's the
point of having built the small version first.

## Things this prototype deliberately ignores

Multi-language content, per-user conversation persistence across sessions,
token-cost metering, channel-specific webhook idempotency, and rich message
templates. All real concerns for a production system, and all orthogonal to
the RAG core you've been running here.

---

Back to the [tutorial index](README.md)
