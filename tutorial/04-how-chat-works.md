# 4. How chat works (`chat.py`)

Every question you type goes through four steps: **embed → rank → assemble →
generate**.

## Step 1: Embed the question

The question is embedded with the same model and dimensions as the documents,
but with `task_type="RETRIEVAL_QUERY"` (the other half of the asymmetric pair
explained in chapter 3).

## Step 2: Rank every chunk

```python
scored = [
    {**entry, "score": cosine_similarity(q_vec, entry["embedding"])}
    for entry in store["entries"]
]
scored.sort(key=lambda e: e["score"], reverse=True)
top = [e for e in scored[:TOP_K] if e["score"] >= MIN_SIMILARITY]
```

This is a brute-force scan — the question's vector is compared against every
stored chunk. At this scale it takes microseconds. (pgvector does the same
thing with an index once you have thousands of chunks.)

Two guards work together (defaults live in `config.py`, overridable via
`.env` — see `TOP_K`/`MIN_SIMILARITY` in `.env.example`):

- **`TOP_K = 4`** caps how much context we send (cost + focus).
- **`MIN_SIMILARITY = 0.35`** is the "actually relevant" floor. If *nothing*
  clears it, the code doesn't call the LLM at all — it returns a fixed "I
  don't have that information" reply. **The safest answer is the one the
  model never gets to improvise.**

## Step 3: Assemble the grounded prompt

The retrieved chunks are pasted into a labelled CONTEXT block:

```
CONTEXT:
[1] (source: example.md)
Example Topic — How should I format my files?
...

---

[2] (source: example.md)
...

QUESTION: how do I format my knowledge base files?
```

## Step 4: Generate under rules

The model is `gemini-2.5-flash` with `temperature=0.2` (low randomness — we
want consistent, factual replies, not creative ones) and this system
instruction:

> - Answer ONLY from the CONTEXT block provided with each question.
> - If the context does not contain the answer, say you don't have that
>   information. Never guess.
> - Quote facts, numbers, and figures exactly as written in the context.
>   Never invent, estimate, or round them.
> - Never promise anything that is not stated in the context.

Each rule exists because of a specific failure mode: models pad answers with
trained "general knowledge" (rule 1), guess rather than refuse (rule 2), round
or convert numbers (rule 3), and mirror a user's request with agreeable
improvisation (rule 4).

## Conversation memory

The script uses `client.chats.create(...)` once and `chat.send_message(...)`
per turn, so **Gemini sees the whole conversation history** — follow-ups like
"and what about the second point?" resolve against the previous answer. Note
what this implies: earlier retrieved context stays in the history, so a
follow-up can be answered from chunks retrieved two questions ago. Handy in a
prototype; in production you'd cap history length to control token cost.

## The sources line

```
[retrieved: example.md (0.68)]
```

This is your debugging window. Read it as: *what did retrieval find, and how
confident was it?* Chapter 6 shows how to diagnose bad answers with it — the
first question is always "did the right chunk get retrieved?", because if it
didn't, no prompt engineering will fix the answer.

---

Next: [5. Writing knowledge-base files](05-writing-kb-files.md)

