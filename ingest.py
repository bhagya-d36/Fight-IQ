"""ingest.py — reads every .md file in ./knowledge-base, splits it into chunks,
embeds each chunk locally with sentence-transformers, and saves everything to
vector-store.json.

Re-run this whenever you edit the knowledge base:  python ingest.py
Unchanged files are skipped and their existing chunks/embeddings are reused —
only new or edited files are re-embedded.

Preview chunking without embedding:  python ingest.py --dry-run
Re-embed everything from scratch:    python ingest.py --force
"""

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import config
import embeddings
from net_fix import prefer_ipv4

prefer_ipv4()

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge-base"
STORE_FILE = BASE_DIR / "vector-store.json"
STORE_VERSION = 3

DRY_RUN = "--dry-run" in sys.argv
FORCE = "--force" in sys.argv


def _overlap_tail(body: str, overlap: int) -> str:
    """Last ~overlap chars of body, trimmed to start at a clean boundary."""
    if overlap <= 0 or len(body) <= overlap:
        return body.strip()
    tail = body[-overlap:]
    for sep in ("\n\n", "\n", ". ", " "):
        idx = tail.find(sep)
        if idx != -1:
            trimmed = tail[idx + len(sep) :].strip()
            if trimmed:
                return trimmed
    return tail.strip()


def chunk_markdown(
    file_name: str,
    text: str,
    max_chars: int = config.MAX_CHUNK_CHARS,
    overlap: int = config.MAX_CHUNK_OVERLAP,
) -> list[str]:
    """Split a markdown document into chunks.

    Strategy: split on "## " headings so each chunk is one coherent topic,
    then split any oversized section on blank lines, carrying a small tail
    of each packed chunk into the next so a fact split across a packing
    boundary isn't orphaned. Every chunk is prefixed with its document title
    + heading so it stays self-describing after retrieval (the LLM never
    sees the rest of the file).
    """
    overlap = min(overlap, max_chars // 2)  # clamp so packing always makes forward progress
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
        # Oversized section: pack paragraphs greedily up to the limit, carrying
        # a small overlap tail into the next chunk so a fact split across the
        # boundary isn't orphaned.
        current = ""
        for para in re.split(r"\n\s*\n", body):
            if current and len(current) + len(para) > max_chars:
                chunks.append(prefix + current.strip())
                current = _overlap_tail(current, overlap) + "\n\n"
            current += para + "\n\n"
        if current.strip():
            chunks.append(prefix + current.strip())
    return chunks


def file_sha256(text: str) -> str:
    """Hash file content (not mtime — cloud-synced folders touch mtimes on their own)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_reusable_store(force: bool) -> dict | None:
    """Return the existing store if its chunks can be reused as a starting
    point for incremental re-embedding, else None (triggers a full rebuild).
    """
    if force or not STORE_FILE.exists():
        return None
    try:
        store = json.loads(STORE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if (
        store.get("version") != STORE_VERSION
        or store.get("model") != config.EMBEDDING_MODEL
        or store.get("chunkChars") != config.MAX_CHUNK_CHARS
        or store.get("chunkOverlap") != config.MAX_CHUNK_OVERLAP
    ):
        return None
    return store


def embed_entries(entries: list[dict]) -> None:
    """Embed `entries` in place, adding an "embedding" key to each."""
    if not entries:
        return

    print(f"Embedding {len(entries)} chunk(s) with {config.EMBEDDING_MODEL}...")
    BATCH = 50
    for i in range(0, len(entries), BATCH):
        batch = entries[i : i + BATCH]
        vectors = embeddings.embed_texts([e["text"] for e in batch])
        for entry, vector in zip(batch, vectors):
            entry["embedding"] = vector
        print(f"  {min(i + BATCH, len(entries))}/{len(entries)}")


def main() -> None:
    if not KB_DIR.is_dir():
        sys.exit(f"Knowledge base folder not found: {KB_DIR}")
    files = sorted(KB_DIR.glob("*.md"))
    if not files:
        sys.exit("No .md files in knowledge-base/. Add some and re-run.")

    old_store = load_reusable_store(FORCE)
    old_hashes: dict[str, str] = old_store["files"] if old_store else {}
    old_entries_by_source: dict[str, list[dict]] = {}
    for entry in (old_store["entries"] if old_store else []):
        old_entries_by_source.setdefault(entry["source"], []).append(entry)

    entries: list[dict] = []
    to_embed: list[dict] = []
    file_hashes: dict[str, str] = {}
    reused_files = 0
    changed_files = 0

    for path in files:
        text = path.read_text(encoding="utf-8")
        digest = file_sha256(text)
        file_hashes[path.name] = digest

        if digest == old_hashes.get(path.name):
            reused = old_entries_by_source.get(path.name, [])
            entries.extend(reused)
            reused_files += 1
            print(f"{path.name}: unchanged ({len(reused)} chunk(s) reused)")
        else:
            chunks = chunk_markdown(path.name, text)
            new_entries = [{"source": path.name, "text": chunk} for chunk in chunks]
            entries.extend(new_entries)
            to_embed.extend(new_entries)
            changed_files += 1
            print(f"{path.name}: changed -> {len(chunks)} chunk(s) to embed")

    removed = sorted(set(old_hashes) - {p.name for p in files})
    for name in removed:
        print(f"removed: {name}")

    if DRY_RUN:
        for entry in entries:
            print(f"\n--- [{entry['source']}] {len(entry['text'])} chars ---")
            print("\n".join(entry["text"].splitlines()[:3]))
        print(f"\nDry run: {len(entries)} chunks total. No embeddings created.")
        return

    embed_entries(to_embed)

    STORE_FILE.write_text(
        json.dumps(
            {
                "version": STORE_VERSION,
                "model": config.EMBEDDING_MODEL,
                "dim": embeddings.dimension(),
                "chunkChars": config.MAX_CHUNK_CHARS,
                "chunkOverlap": config.MAX_CHUNK_OVERLAP,
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "files": file_hashes,
                "entries": entries,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(
        f"Done. {len(files)} files: {changed_files} re-embedded, {reused_files} reused, "
        f"{len(removed)} removed -> {len(entries)} chunks."
    )


if __name__ == "__main__":
    main()
