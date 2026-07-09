"""ingest.py — reads every .md file in ./knowledge-base, splits it into chunks,
embeds each chunk with Gemini, and saves everything to vector-store.json.

Re-run this whenever you edit the knowledge base:  python ingest.py
Preview chunking without API calls:                python ingest.py --dry-run
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import config
from net_fix import prefer_ipv4

prefer_ipv4()

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge-base"
STORE_FILE = BASE_DIR / "vector-store.json"

DRY_RUN = "--dry-run" in sys.argv

if not os.environ.get("GEMINI_API_KEY") and not DRY_RUN:
    sys.exit("Missing GEMINI_API_KEY. Create a .env file (see .env.example).")


def chunk_markdown(file_name: str, text: str, max_chars: int = config.MAX_CHUNK_CHARS) -> list[str]:
    """Split a markdown document into chunks.

    Strategy: split on "## " headings so each chunk is one coherent topic,
    then split any oversized section on blank lines. Every chunk is prefixed
    with its document title + heading so it stays self-describing after
    retrieval (the LLM never sees the rest of the file).
    """
    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    doc_title = title_match.group(1).strip() if title_match else file_name
    sections = re.split(r"^(?=##\s)", text, flags=re.MULTILINE)  # keep headings
    chunks: list[str] = []

    for section in sections:
        body = section.strip()
        if not body:
            continue
        # Skip sections that are only a title line with no content (e.g. the
        # "# Doc Title" intro before the first "##" heading).
        if not re.sub(r"^#{1,6}\s+.+$", "", body, flags=re.MULTILINE).strip():
            continue
        heading_match = re.search(r"^##\s+(.+)$", body, re.MULTILINE)
        heading = heading_match.group(1).strip() if heading_match else ""
        prefix = f"{doc_title} — {heading}\n\n" if heading else f"{doc_title}\n\n"

        if len(body) <= max_chars:
            chunks.append(prefix + body)
            continue
        # Oversized section: pack paragraphs greedily up to the limit.
        current = ""
        for para in re.split(r"\n\s*\n", body):
            if current and len(current) + len(para) > max_chars:
                chunks.append(prefix + current.strip())
                current = ""
            current += para + "\n\n"
        if current.strip():
            chunks.append(prefix + current.strip())
    return chunks


def main() -> None:
    if not KB_DIR.is_dir():
        sys.exit(f"Knowledge base folder not found: {KB_DIR}")
    files = sorted(KB_DIR.glob("*.md"))
    if not files:
        sys.exit("No .md files in knowledge-base/. Add some and re-run.")

    entries: list[dict] = []
    for path in files:
        chunks = chunk_markdown(path.name, path.read_text(encoding="utf-8"))
        print(f"{path.name}: {len(chunks)} chunk(s)")
        entries.extend({"source": path.name, "text": chunk} for chunk in chunks)

    if DRY_RUN:
        for entry in entries:
            print(f"\n--- [{entry['source']}] {len(entry['text'])} chars ---")
            print("\n".join(entry["text"].splitlines()[:3]))
        print(f"\nDry run: {len(entries)} chunks total. No embeddings created.")
        return

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    print(f"Embedding {len(entries)} chunks with {config.EMBEDDING_MODEL}...")
    BATCH = 50
    for i in range(0, len(entries), BATCH):
        batch = entries[i : i + BATCH]
        res = client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=[e["text"] for e in batch],
            config=types.EmbedContentConfig(
                output_dimensionality=config.EMBEDDING_DIM,
                task_type="RETRIEVAL_DOCUMENT",
            ),
        )
        for entry, embedding in zip(batch, res.embeddings):
            entry["embedding"] = list(embedding.values)
        print(f"  {min(i + BATCH, len(entries))}/{len(entries)}")

    STORE_FILE.write_text(
        json.dumps(
            {
                "model": config.EMBEDDING_MODEL,
                "dim": config.EMBEDDING_DIM,
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "entries": entries,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"Done. Wrote {len(entries)} chunks to vector-store.json")


if __name__ == "__main__":
    main()
