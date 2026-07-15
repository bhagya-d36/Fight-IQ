# 6. Tuning and troubleshooting

## The knobs

Defaults live in `config.py`; override any of them via `.env` (see
`.env.example`) without touching code:

| Knob | Default | Effect of raising it | Effect of lowering it |
|---|---|---|---|
| `TOP_K` | 4 | More context per answer; better for questions spanning topics; more tokens/cost | Tighter, cheaper prompts; may miss a needed chunk |
| `MIN_SIMILARITY` | 0.35 | Stricter: more "I don't know", fewer wrong-context answers | Looser: answers more questions, higher risk of answering from a barely-related chunk |
| `MAX_CHUNK_CHARS` | 1500 | Bigger chunks: more context each, but blurrier vectors | Sharper matching, but facts may split apart |
| `temperature` (in `rag.py`, not env-driven) | 0.2 | More varied phrasing | More deterministic (0 = most rigid) |
| `EMBEDDING_DIM` | 768 | Marginally better matching, bigger store | Smaller store; don't go below ~256 |

Changing `MAX_CHUNK_CHARS`, the embedding model, or `EMBEDDING_DIM` requires
`python ingest.py --force` (see chapter 3). `TOP_K`/`MIN_SIMILARITY` are
query-time only — just restart `chat.py`/`server.py` after editing `.env`.

## Diagnosing a bad answer — always in this order

**First, read the `[retrieved: ...]` line.** It splits every problem into one
of two categories:

- **Wrong/no chunks retrieved** → a *retrieval* problem. Fix the KB writing or
  the knobs. No prompt change will help.
- **Right chunks retrieved, wrong answer** → a *generation* problem. Fix the
  system instruction or lower the temperature.

## Symptom → cause → fix

### "I don't have that information" — but the answer IS in the KB

- Check retrieval scores: if the right file appears at 0.30–0.34, it was cut by
  the threshold. Either lower `MIN_SIMILARITY` slightly, or (better) reword the
  section heading closer to how people actually ask (chapter 5, rule 4).
- If the right file doesn't appear at all: the fact is probably buried in a
  section about something else. Give it its own `##` section and re-ingest.
- Did you edit the KB and forget `python ingest.py`? (The most common cause.)

### Answers cite the right file but miss the specific detail

The chunk containing the detail is probably over 1,500 chars and got split from
its context. Run `python ingest.py --dry-run`, find the section, split it into
smaller self-contained `##` sections.

### Bot answers confidently about things NOT in the KB

- Check whether a loosely-related chunk cleared the threshold and the model
  over-extrapolated from it → raise `MIN_SIMILARITY` toward 0.45.
- Remember history: earlier retrieved context is still visible to the model, so
  a follow-up may legitimately use chunks from a previous question.

### Two similar topics get mixed up

Both chunks retrieve together and the model blends them. Make each section
name its subject explicitly in every line ("The annual plan costs...", not "It
costs..."), so the model can't cross-wire them.

### `429` / rate-limit or quota errors

Free-tier Gemini limits requests per minute. Each chat turn = 1 embedding call
+ 1 generation call; ingest = 1 call per 50 chunks. The client (`rag.make_client()`)
already retries 429s and transient network errors a few times with backoff, so
brief blips resolve on their own; a persistent `429` means you need to wait
longer or enable billing on the key. If ingest fails partway through, just
re-run it — unaffected files are reused from the cache (chapter 3), so you
only pay for the files still pending.

### Answers are slow

Normal: one embedding round-trip + one generation round-trip per question.
`gemini-2.5-flash` also spends tokens "thinking" by default; for snappier
replies you can add `thinking_config=types.ThinkingConfig(thinking_budget=0)`
to the chat config in `chat.py`.

## A 5-minute eval habit

Keep a list of ~15 real questions you expect people to ask (mix: things
clearly in the KB, edge cases, things NOT in the KB, different phrasings).
After every KB edit or knob change, run through them and note wrong answers.
This tiny manual eval gives you visibility into what the LLM actually said —
the same thing a production system would log per-turn at scale.

---

Next: [7. Path to production](07-path-to-production.md)

