# 5. Writing knowledge-base files

The knowledge base is just the `knowledge-base/` folder. Add or edit `.md`
files, run `python ingest.py`, done. But *how* you write them determines
whether retrieval finds the right chunk. The rules below all follow from one
fact: **each `##` section becomes one chunk, and the LLM only ever sees
retrieved chunks — never the whole file.**

## Rule 1 — One file per topic, meaningful filename

The filename appears in the `[retrieved: ...]` line and in the prompt as the
source. `pricing.md` beats `doc2.md`.

## Rule 2 — Start with `# Title`, then one `##` per distinct fact

The chunker splits on `##`. A section should answer exactly one question. If
you find yourself writing "Also, ..." — that's usually a new section.

## Rule 3 — Make every section self-contained

The reader of a chunk (the LLM) has no access to the surrounding file. Pronouns
and back-references break silently:

**Bad** (depends on the previous section):

```markdown
## Annual plan
Same as monthly, but billed once a year.
```

**Good:**

```markdown
## Annual plan pricing
The annual plan is $290/year, billed once a year (equivalent to ~$24/month).
```

If "Same as monthly" is retrieved alone, the model either refuses or — worse —
guesses what "monthly" was.

## Rule 4 — For FAQs, make the heading the literal question

Questions embed close to questions. `## Can I cancel my subscription?` will
match "how do I cancel" far better than `## Cancellation policy`. Write
headings in the voice of the person asking, not the person answering.

## Rule 5 — Keep sections comfortably under 1,500 characters

Longer sections are auto-split on paragraph breaks, which can separate a rule
from its exception. If a section is growing, split it into more `##` sections
yourself — you'll choose better boundaries than the character counter.

## Rule 6 — State facts with their units and conditions inline

Every number with its unit and condition in the same sentence:
`Storage is capped at 50GB per user on the free tier.` Never put the unit in
one section and the number in another — they may not be retrieved together.

## Rule 7 — Don't duplicate facts across files

If a fact lives in two files and you update one, retrieval may surface the
stale one. One fact, one home.

## Anti-pattern gallery

| Anti-pattern | Why it fails |
|---|---|
| One giant `## Everything` section | Becomes one blurry vector; matches all questions weakly |
| Tables split across sections | Header row and data rows end up in different chunks |
| Marketing fluff ("world-class service!") | Wastes chunk space; embeds close to nothing people ask |
| Relative dates ("starting next month") | Chunk outlives the meaning; use absolute dates |
| Internal notes mixed into public docs | Retrieval can surface them to users verbatim |

## Update workflow

1. Edit or add `.md` files in `knowledge-base/`.
2. `python ingest.py --dry-run` — check the chunks look like sensible units.
3. `python ingest.py`.
4. In `python chat.py`, ask 2–3 questions a real user would actually phrase,
   and check the `[retrieved: ...]` line hits the file you just edited.

---

Next: [6. Tuning and troubleshooting](06-tuning-and-troubleshooting.md)

