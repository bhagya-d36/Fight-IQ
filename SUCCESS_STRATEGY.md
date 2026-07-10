# FightIQ — Success Strategy

*A plan for taking this RAG chatbot from working prototype to a public portfolio piece that earns recognition. Written July 2026. Grounded in what the repo actually is today: Gemini embeddings + chat, hand-curated UFC markdown knowledge base, JSON vector store with incremental ingest, FastAPI web UI with streaming, offline test suite.*

---

## 0. The thesis (read this first)

**FightIQ's real product is trust in a domain where facts expire weekly.**

Almost every RAG demo on the internet is "chat with a static PDF." MMA is the opposite: champions, rankings, records, and event cards change every single weekend. Generic chatbots (including ChatGPT) confidently give stale or invented answers here — wrong champion, made-up records, events that never happened. That volatility is not your problem; **it's your showcase**. A bot that says *"Ciryl Gane is the interim heavyweight champion — per champions.md, current as of July 5, 2026"* — and can prove it with a citation and an as-of date — demonstrates the exact engineering problem (grounding + freshness + honest refusal) that production RAG teams get paid to solve.

Two audiences, one build:

| Audience | What they get | What wins them |
|---|---|---|
| MMA fans (users) | Answers with receipts and dates, not confident guesses | Accuracy on "current champ / next event / fighter record" queries |
| Devs & recruiters (recognition) | A from-scratch RAG with a real eval harness on volatile data | The eval numbers, the freshness pipeline, the honest failure handling |

Everything below serves one of those two.

---

## 1. Positioning — why anyone would use this

### The competitive landscape (researched July 2026)

- **Existing MMA AI products are ~all prediction/betting tools**: [MMA-AI.net](https://mma-ai.net/), [MMAmodel.ai](https://mmamodel.ai/), [Agent MMA](https://agentmma.com/), [UFC Predictor](https://ufc-predictor.com/), [MMA Fight Sim](https://mmafightsim.com/), the MMA AI iOS app. They compete on pick accuracy and ROI claims. None are Q&A/knowledge products; none cite sources.
- **Stat sites** (UFCStats, Tapology, Sherdog, ESPN MMA) have the data but you have to *navigate* to it — no natural-language layer, no synthesis across pages.
- **Generic LLMs** answer MMA questions fluently but are stale (training cutoff) and hallucinate records/results — the single most common complaint in MMA subreddits about asking ChatGPT fight questions.

### The gap FightIQ fills

**"The MMA answer engine with receipts."** From a user's point of view, the three moments where FightIQ beats every alternative:

1. **Currency questions** — "Who's the heavyweight champ right now?" / "Who does Ulberg defend against?" ChatGPT is stale, Google returns a page to read, FightIQ answers in one sentence *with an as-of date and source*.
2. **Rules & scoring disputes** — "Was that knee illegal?" "Why did he lose a point?" "How does the 10-point must system work?" Fans argue about this constantly; the unified rules are public, stable, and perfect RAG material. This is a permanently-fresh content moat that needs almost zero maintenance.
3. **Honest refusal** — ask something the KB doesn't cover and it says so instead of inventing. Counterintuitively, this is a *feature to advertise*, not hide: "the MMA bot that says 'I don't know' instead of lying to you."

### What NOT to position as

- ❌ **Fight predictions** — crowded, betting-adjacent (credibility risk for a portfolio), and RAG is the wrong tool for it.
- ❌ **"ChatGPT for MMA"** — invites direct comparison to a model 1000× bigger; you lose.
- ❌ **A stats database replacement** — Tapology/UFCStats own that; you're the language layer, not the ledger.

### One-line pitches (use these verbatim)

- To users: *"Ask anything about the UFC — every answer cites its source and tells you how fresh it is."*
- To devs/HN: *"I built a RAG system for a domain where the ground truth changes every Saturday night — here's how I keep it from hallucinating."*

---

## 2. Technical bar — what "production-grade" means at portfolio scale

You don't need enterprise infra. You need the *behaviors* of production RAG, demonstrable in a 2-minute demo. Current state → target:

### 2.1 Accuracy & groundedness

| Behavior | Status | Target |
|---|---|---|
| Answers only from context | ✅ system prompt in `rag.py` | Keep |
| Refuses on no retrieval hit | ✅ `MIN_SIMILARITY` gate + `NO_MATCH_ANSWER` | Keep, but tune threshold *with evals*, not by feel |
| Exact figures, no rounding | ✅ prompted | **Verify with eval set** — prompts are claims, evals are proof |
| Faithfulness measured | ❌ | **RAGAS-style eval, faithfulness ≥ 0.85 floor** (industry-standard bar; below 0.70 = significant hallucination) |
| Golden Q&A dataset | ❌ | **75–150 hand-verified Q&A pairs** — the single highest-leverage artifact in this whole document (see §2.4) |

### 2.2 Citations (upgrade from prompt-level to UI-level)

Today the model *mentions* sources in prose ("per rankings.md"). Production-grade is **structural citations**: the API already returns `hits` with `source` and `score` — render them as clickable chips under each answer in `web/`, expanding to show the exact retrieved chunk. This is cheap (the data is already flowing through `server.py`) and it's the #1 visual credibility signal in a demo. Add a "knowledge base last updated: <date>" line to the UI footer — make freshness visible, not just prompted.

### 2.3 Latency

- Streaming already exists (`/api/ask/stream`) — good, that's the hard part.
- Bar: **first token < ~1.5s, full answer p95 < 8s**. Measure it (log per-stage timings: embed call, similarity scan, first token) and *publish the numbers in the README*. A latency table is a production signal almost no portfolio RAG has.
- The pure-Python cosine scan over `vector-store.json` is fine to ~5–10k chunks. Don't swap in a vector DB preemptively (see §4 pivots) — but *say in the README that you know where the ceiling is*. Knowing your scaling limits reads as senior; premature Pinecone reads as tutorial-following.

### 2.4 The eval harness (the centerpiece)

Build `evals/` with a golden dataset covering five query classes:

1. **Stable facts** — "How many weight classes are there?" (should always be right)
2. **Volatile facts** — "Who is the current lightweight champion?" (right *and* freshness-annotated)
3. **Temporal traps** — questions whose answer changed recently ("Is Jon Jones the heavyweight champ?" → *no, and here's the history*). This class is your differentiator; no PDF-chat project has it.
4. **Out-of-scope** — "Who won the boxing match last night?" / "What's a good parlay?" (must refuse)
5. **Opinion/ambiguous** — "Who's the GOAT?" "Was that decision robbery?" (must present sourced perspectives or decline to rule, never assert opinion as fact)

For each: score **retrieval** (did the right chunk make top-K? → recall@K, MRR) separately from **generation** (faithfulness, answer correctness). Run via `pytest`-style script; print a scorecard table; commit the scorecard to the README. Frameworks: [RAGAS](https://qaskills.sh/blog/ragas-rag-evaluation-metrics-complete-guide) is the standard, but at this scale a hand-rolled LLM-as-judge script is *more* impressive (consistent with the project's no-framework ethos) — cite RAGAS's metric definitions, implement them yourself.

**Then use it**: every chunking/threshold/prompt change becomes "faithfulness went 0.79 → 0.91." That sentence is the whole portfolio story.

### 2.5 Handling opinion-based queries (MMA-specific)

MMA queries are disproportionately opinion-shaped ("Was Khabib ducking Tony?"). Define explicit behavior: detect opinion-class questions → answer with *sourced framing* ("Analysts point to X; the record shows Y") and never a verdict. Encode this in the system prompt *and* the eval set. Recruiters specifically probe how projects handle ambiguity — having a designed answer (not an accident) is rare.

---

## 3. Data & knowledge base

### 3.1 Sourcing strategy — the licensing-safe stack

| Source | License/risk | Use for |
|---|---|---|
| **Wikipedia / Wikidata** | CC BY-SA — safe to use with attribution; updated within *minutes-to-hours* of major UFC events by fan editors | **Backbone.** Champions, event results, fighter bios, rankings snapshots |
| **Unified Rules of MMA** (ABC), UFC rankings methodology, athletic commission docs | Public regulatory/rules text | Rules & scoring corpus — zero staleness risk |
| **UFCStats.com** | No API; widely scraped ([community scrapers](https://github.com/FritzCapuyan/ufc-api)) but ToS-gray | Historical per-fight stats *if* you go structured (§4). Don't redistribute raw dumps |
| **Tapology** | Aggressive anti-scraping, IP bans | **Avoid** |
| **Sherdog** | Scraped by many projects, ToS-gray | Avoid as primary; not worth the risk for a public portfolio piece |
| **[SportsDataIO MMA API](https://sportsdata.io/mma-ufc-api)** and similar | Commercial, paid | Only if this becomes a product; overkill for portfolio |
| **Your own prose** | Yours | Explainers, rules summaries, FAQ phrasing — this hand-curation is genuinely why answers read well; keep it |

**Rule for a public project: everything in `knowledge-base/` must be either CC-licensed, public-domain, or written by you.** Scraped ToS-gray content is fine for private ML experiments; in a portfolio repo with your name on it, it's a liability a reviewer can spot in one click.

### 3.2 Update cadence & freshness pipeline

MMA has a natural clock — **events are Saturdays**. Design around it:

| Data class | Changes | Cadence |
|---|---|---|
| Rules, scoring, weight classes, history | ~never | On write, then annually |
| Fighter bios, notable-fighter profiles | Monthly-ish | Monthly touch-up |
| Rankings | Tuesdays after events | Weekly |
| Champions, event results, upcoming cards | Every event weekend | **Sunday/Monday after each event** |

Ship it in three maturity stages (each stage is a README-worthy milestone):

1. **Now (manual, honest)**: Add a `last_updated` field per KB file (front-matter), surface the *oldest volatile file's* date in the UI. You already prompt the model to caveat staleness — make the data structural.
2. **Next (verified freshness)**: A `freshness_check.py` script — for each volatile file, pull the relevant Wikipedia page/Wikidata entity, diff against KB claims, and *flag* discrepancies for you to review (human-in-the-loop, not auto-write). Run it Sunday nights. This script is a better portfolio artifact than a fancier retriever.
3. **Later (automated)**: GitHub Action on a Sunday-night cron: fetch → regenerate volatile `.md` sections → `ingest.py` (incremental hashing already makes this cheap) → commit. Fully automated freshness with the diff history *visible in git log* — a public, auditable freshness pipeline. Very few RAG portfolios have this; it's the strongest possible answer to "how do you handle stale data?"

### 3.3 Avoiding stale/incorrect answers — defense in depth

You already have layers 1–3; add 4–6:

1. ✅ Similarity gate → refuse rather than stretch
2. ✅ Prompt: quote figures exactly, caveat volatile facts
3. ✅ KB files self-describe their currency ("current as of July 2026")
4. ➕ Structural `last_updated` per chunk, injected into context (`[source: champions.md, updated 2026-07-05]`) so the model can caveat *specifically*, and the UI can show it
5. ➕ Temporal-trap evals (§2.4) so staleness regressions are *caught*, not discovered by users
6. ➕ Freshness checker (§3.2) so staleness is *prevented*

---

## 4. Pivots & scope risks

### Likely failure points and pre-planned responses

**Risk 1: Hallucination on numeric stats (records, reach, strike counts).**
Embedding numbers in prose chunks and hoping the LLM copies them correctly is the weakest link in every RAG system. *Pivot that similar projects made successfully*: split the KB into **prose (RAG) + structured facts (exact lookup)** — a small SQLite/JSON table of fighter records the model queries via one function call, with RAG for everything narrative. "I moved numeric facts out of the embedding path because evals showed X% error" is a *great* story. Trigger: eval set shows figure errors > ~2%.

**Risk 2: Hand-curated KB doesn't scale / goes stale the week you stop maintaining it.**
This kills most fan-made projects (dead data = dead project, visibly). Mitigation: the freshness pipeline (§3.2) — automation stage 3 makes the project self-sustaining. Trigger for pivot: if you find yourself skipping two consecutive event-weekend updates, *stop expanding coverage and automate what exists*.

**Risk 3: Niche too narrow (UFC-only Q&A has modest daily utility).**
Don't widen to all combat sports (data burden ×5). Instead deepen into the moments of peak demand: **fight-week mode** — for the upcoming card, pre-built tale-of-the-tape, storylines, "what's at stake" per bout. Usage of MMA content spikes 10× on event weekends; meet the spike. Trigger: launch feedback says "cool but I wouldn't come back."

**Risk 4: Niche too broad for the *real* goal (recognition).**
If user growth stalls, remember the recognition audience is developers. *Pivot that works*: reframe the repo as **"a case study in RAG on volatile data"** — the eval harness, freshness pipeline, and write-up become the product; MMA becomes the demo domain. Several well-received Show HN RAG posts were exactly this shape (the write-up outperformed the app). This pivot costs nothing because you build the same artifacts either way — that's why this strategy front-loads evals.

**Risk 5: Data licensing complaint (UFC/Zuffa is litigious about footage & marks).**
You're using facts (not copyrightable) + CC text + your prose, no UFC imagery/logos/footage — low risk. Keep it that way: no UFC logo in the UI, name stays "FightIQ," add a "not affiliated with UFC/Zuffa" footer line. Trigger for action: any takedown contact → comply immediately, write up the lesson (even that makes good content).

**Risk 6: Gemini API cost/quota if the demo gets HN traffic.**
Free-tier quota dies on launch day; a dead demo is worse than no demo. Pre-launch: per-IP rate limit in `server.py`, cache embeddings for repeated questions (trivial: hash question → cached answer for identical strings), set a billing cap, and have a "demo is at capacity, here's a 90-sec video" fallback path.

---

## 5. Launch & recognition path

### Where and in what order

**Phase 0 — quiet polish (before any post):** README with demo GIF, live demo URL (Render/Railway/Fly free tier), eval scorecard published, rate limiting on. A launch post is unrepeatable; don't spend it early.

**Phase 1 — Show HN** (highest credibility-per-viewer for this audience):
- Title formula: direct + technical hook + the volatile-data angle. E.g. **"Show HN: A RAG chatbot for UFC facts that cites sources and knows when it's stale"** — not "AI-powered MMA assistant" (marketing language is an instant turnoff on HN).
- First comment: why you built it, the no-framework design choice, the eval numbers, one honest limitation, invite feedback. Personal + technical, zero marketing.
- Demo must work **without signup** — HN users bounce at any gate.
- Post Mon–Wed, morning US time. Respond to every comment for the first 4–6 hours.
- HN loves: from-scratch implementations (no LangChain — *lead with this*), honest eval numbers including failures, "I don't know" as a feature.

**Phase 2 — Reddit, one subreddit at a time:**
- **r/Rag** and **r/LocalLLaMA** — the eval-harness/freshness-pipeline angle; these communities give substantive technical feedback.
- **r/MMA** — *only* with a fan-first frame ("I built a free tool that answers UFC questions with sources — roast its answers") and only after checking self-promo rules; time it to a fight week. Fans stress-testing it = free eval data.
- **Avoid r/MachineLearning** for a project post — it's research-oriented and will remove or ignore it.

**Phase 3 — the write-up (the compounding asset):**
A technical blog post: **"RAG when the ground truth changes every Saturday: evals, freshness, and honest refusals."** Structure: problem → why static-PDF RAG patterns fail here → eval harness with real before/after numbers → freshness pipeline → what didn't work. Cross-post to dev.to/Medium, thread-ify for X, single-image (eval scorecard) post for LinkedIn. **For recruiters, this post is worth more than the app** — it proves you can reason about systems, not just wire them.

**Skip Product Hunt** — it rewards polished consumer products with launch-day audiences; a portfolio RAG tool gets buried. Spend that effort on the write-up.

### What makes these posts land (pattern across successful launches)

1. A specific, falsifiable claim ("faithfulness 0.91 on a 120-question golden set") beats any adjective.
2. One honest limitation stated up-front buys credibility for every other claim.
3. Interactivity: reviewers must be able to break it in 10 seconds of clicking. Every field report on portfolio review says the same thing: **a clickable demo outweighs the code**.
4. The story of a *decision* ("I moved numbers out of the embedding path because…") beats a list of features.

---

## 6. Credibility signals — what reviewers/recruiters actually check

In rough order of what gets looked at first:

1. **README first screen**: one-sentence pitch → demo GIF (10–20s: ask a currency question, see the cited, dated answer) → live demo link → eval scorecard table. All four above the fold.
2. **Live demo** that works without setup. Streaming + citation chips + freshness date visible.
3. **Eval scorecard in the README** — the rarest and strongest signal. Include the *methodology* (query classes, judge setup, dataset size) and the *failures* ("stable facts 98%, temporal traps 84% — here's the trap it still falls for"). Reported honest failure = senior signal.
4. **Architecture section with a diagram** and *justified* decisions: why no vector DB (scale honesty), why no LangChain (learning the primitives), why JSON store (inspectability), where each ceiling is. Reviewers probe "why" in interviews — pre-answer them.
5. **Tests + CI badge**: offline pytest suite already exists — add GitHub Actions (`pytest` + `ruff` on push; later, the eval gate with a faithfulness floor so a prompt change that regresses grounding *fails CI* — that one workflow file is a production-engineering signal few portfolios have).
6. **Latency numbers** published (p50/p95, per stage).
7. **Git history that tells a story** — yours already does (incremental ingest → retries → tests → sessions). Keep committing in reviewable units.
8. **The tutorial/ directory** — unusual and good; frame it in the README as "how it works, step by step" — signals communication skill.

---

## 7. Prioritized action plan

**P0 — the proof (do before anything public; ~1–2 weeks of evenings)**
1. Build `evals/`: 75–150 golden Q&A pairs across the 5 query classes (§2.4) → verify: script prints a scorecard; you know your real faithfulness number.
2. Citation chips + KB-freshness date in the web UI (data already in `hits`) → verify: every answer shows clickable sources with the retrieved text.
3. Per-file `last_updated` front-matter, injected into context and shown in UI → verify: volatile answers carry specific dates.
4. Tune `TOP_K`/`MIN_SIMILARITY`/chunking against the eval set → verify: scorecard improves; record before/after.

**P1 — the shine (launch prerequisites; ~1 week)**
5. Deploy the demo publicly; add per-IP rate limiting + billing cap + capacity fallback → verify: stranger can use it from a phone, hammering it doesn't bankrupt you.
6. README overhaul per §6 (pitch, GIF, demo link, scorecard, architecture diagram, latency table) → verify: a friend understands the project in 60 seconds without you talking.
7. GitHub Actions: pytest + ruff badge; stretch: eval gate with faithfulness floor.
8. `freshness_check.py` (stage-2 pipeline, human-in-the-loop) → verify: it catches a deliberately-planted stale claim.

**P2 — the launch (~2 weeks, sequenced)**
9. Sunday-night update ritual through one full event cycle (proves cadence before claiming it).
10. Show HN post + first comment drafted, reviewed, posted Mon–Wed; clear your calendar for comment duty.
11. r/Rag / r/LocalLLaMA posts (eval angle), then r/MMA on a fight week (fan angle).
12. Write and publish the technical post (§5 Phase 3); add to LinkedIn/X with the scorecard image.

**P3 — the compounding (only if traction/energy)**
13. Automated freshness via GitHub Actions cron (stage 3).
14. Structured-facts lookup for numeric stats if evals show figure errors (§4 Risk 1).
15. Fight-week mode (§4 Risk 3).

**Explicit non-goals** (write them down so scope can't creep): fight predictions, betting content, all-combat-sports coverage, vector DB migration before ~5k chunks, any framework rewrite.

---

## Appendix: key sources

- Launch mechanics: [How to do a successful HN launch](https://www.lucasfcosta.com/blog/hn-launch), [Ask HN: tips for a good Show HN](https://news.ycombinator.com/item?id=40563762), [HN posting guide](https://syften.com/blog/hacker-news-marketing/)
- Evals: [RAG evaluation metrics guide (RAGAS)](https://qaskills.sh/blog/ragas-rag-evaluation-metrics-complete-guide), [RAG evaluation 101 — recall@K to faithfulness](https://langcopilot.com/posts/2025-09-17-rag-evaluation-101-from-recall-k-to-answer-faithfulness), [RAG evaluation metrics 2026](https://futureagi.com/blog/rag-evaluation-metrics-2025/)
- Portfolio signals: [Ultimate guide to AI engineering portfolios](https://www.dataexpert.io/blog/ultimate-guide-ai-engineering-portfolios), [ML portfolio projects that get you hired](https://www.interviewnode.com/post/ml-engineer-portfolio-projects-that-will-get-you-hired-in-2025)
- Data landscape: [SportsDataIO MMA API](https://sportsdata.io/mma-ufc-api) (commercial reference), [community UFC scraper](https://github.com/FritzCapuyan/ufc-api), [Wikimedia Enterprise / Wikidata](https://enterprise.wikimedia.com/project-data/wikidata-api/)
- Competitive scan: [MMA-AI.net](https://mma-ai.net/), [MMAmodel.ai](https://mmamodel.ai/), [Agent MMA](https://agentmma.com/), [MMA Fight Sim](https://mmafightsim.com/)
