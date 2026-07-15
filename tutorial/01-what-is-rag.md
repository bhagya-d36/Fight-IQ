# 1. What is RAG, and why you need it

## The problem

Gemini knows nothing about your business, product, or documents. Ask it "how
much does the Pro plan cost?" and it will either admit ignorance or — worse —
**invent a plausible-sounding price**. LLMs are trained to produce fluent
text, not to refuse. For a support bot, a confidently wrong answer is a
disaster: you'd be quoting facts you never set.

Two ways to teach a model your own knowledge:

1. **Fine-tuning** — retrain the model on your data. Expensive, slow to update
   (retrain every time a fact changes), and it still hallucinates.
2. **RAG (Retrieval-Augmented Generation)** — keep your knowledge in ordinary
   files/tables, and at question time *look up* the relevant pieces and paste
   them into the prompt. The model reads them like an open-book exam.

RAG wins whenever your knowledge changes often (prices, policies, docs) and
must be exact. Editing a markdown file and re-running one command is the
whole update process.

## How retrieval actually works: embeddings

Computers can't compare *meaning* directly, so we convert text to numbers.

An **embedding model** (here, `gemini-embedding-001`) turns any piece of text
into a vector — in our setup, a list of **768 numbers**. The model is trained so
that texts with similar meaning get vectors that point in similar directions:

```
"How much is the Pro plan?"          → [0.12, -0.45, 0.88, ...]
"Pro plan: $29 per month"            → [0.10, -0.41, 0.85, ...]   ← close!
"Report a bug within 48 hours"       → [-0.7,  0.02, 0.13, ...]   ← far away
```

"Close" is measured with **cosine similarity**: a score from -1 to 1 for how
aligned two vectors are. Identical meaning ≈ 1.0; unrelated ≈ 0. In this
project, anything above **0.35** against a question is considered relevant.

Crucially, this works **across phrasings** — someone asks "what's the monthly
cost of the paid tier?" and it still lands near the pricing chunk, even though
almost no words overlap. That's what keyword search can't do.

## The full RAG loop

```
                        ONCE, AT INGEST TIME
knowledge-base/*.md → split into chunks → embed each chunk → vector-store.json

                        EVERY QUESTION
user question → embed the question
              → compare against every stored chunk (cosine similarity)
              → take the top 4 matches above the threshold
              → build a prompt:  [system rules] + [those chunks] + [question]
              → Gemini Flash writes the answer USING ONLY the chunks
```

## Grounding: the rule that makes it safe

The system prompt (in `chat.py`) orders the model to answer **only** from the
provided chunks, quote facts and figures **verbatim**, and say "I don't know"
otherwise. This is called **grounding**. In a production system, anything the
model must never get wrong (like a live price) is best fetched from a real
database rather than trusted to the model's cooperation — retrieval finds the
right text, but the numbers themselves shouldn't depend on the model copying
correctly. Defense in depth.

## Why not just paste ALL the knowledge into every prompt?

With a handful of files you could. But: (a) cost — you'd pay for every token
of the whole knowledge base on every message, forever; (b) accuracy — models
get worse at following instructions as prompts grow, and irrelevant text
invites confusion; (c) it doesn't scale as your knowledge base grows into
hundreds of documents. Retrieval keeps prompts small, cheap, and on-topic.

---

Next: [2. Setup and run](02-setup-and-run.md)

