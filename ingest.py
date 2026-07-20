"""ingest.py — reads every .md file in ./knowledge-base, splits it into chunks,
embeds each chunk locally with sentence-transformers, and upserts everything
into a local Chroma vector store (./chroma-store).

Re-run this whenever you edit the knowledge base:  python ingest.py
Unchanged files are skipped and their existing chunks/embeddings are reused —
only new, edited, or deleted files touch the store.

Preview chunking without embedding:  python ingest.py --dry-run
Re-embed everything from scratch:    python ingest.py --force
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import chromadb

import config
import embeddings
from net_fix import prefer_ipv4

prefer_ipv4()

# Some knowledge-base content contains characters outside the Windows
# console's default codepage (e.g. Rakić); without this, --dry-run's preview
# printing crashes on them.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge-base"
CHROMA_DIR = BASE_DIR / "chroma-store"
COLLECTION_NAME = "chunks"
MANIFEST_FILE = CHROMA_DIR / "manifest.json"
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


def load_manifest(force: bool) -> dict | None:
    """Return the existing file-hash manifest if the Chroma collection it
    describes can be reused as a starting point for incremental re-embedding,
    else None (triggers a full rebuild of the collection).
    """
    if force or not MANIFEST_FILE.exists():
        return None
    try:
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if (
        manifest.get("version") != STORE_VERSION
        or manifest.get("model") != config.EMBEDDING_MODEL
        or manifest.get("chunkChars") != config.MAX_CHUNK_CHARS
        or manifest.get("chunkOverlap") != config.MAX_CHUNK_OVERLAP
    ):
        return None
    return manifest


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
    files = sorted(KB_DIR.rglob("*.md"))
    if not files:
        sys.exit("No .md files in knowledge-base/. Add some and re-run.")

    manifest = load_manifest(FORCE)
    old_hashes: dict[str, str] = manifest["files"] if manifest else {}

    file_hashes: dict[str, str] = {}
    to_add: list[dict] = []
    reused_files = 0
    changed_files = 0

    for path in files:
        rel_name = path.relative_to(KB_DIR).as_posix()
        text = path.read_text(encoding="utf-8")
        digest = file_sha256(text)
        file_hashes[rel_name] = digest

        if digest == old_hashes.get(rel_name):
            reused_files += 1
            print(f"{rel_name}: unchanged (reusing existing chunks)")
        else:
            chunks = chunk_markdown(rel_name, text)
            to_add.extend(
                {"id": f"{rel_name}::{i}", "source": rel_name, "text": chunk} for i, chunk in enumerate(chunks)
            )
            changed_files += 1
            print(f"{rel_name}: changed -> {len(chunks)} chunk(s) to embed")

    removed = sorted(set(old_hashes) - {p.relative_to(KB_DIR).as_posix() for p in files})
    for name in removed:
        print(f"removed: {name}")

    if DRY_RUN:
        for entry in to_add:
            print(f"\n--- [{entry['source']}] {len(entry['text'])} chars ---")
            print("\n".join(entry["text"].splitlines()[:3]))
        print(
            f"\nDry run: {len(to_add)} chunk(s) to embed from {changed_files} changed file(s), "
            f"{reused_files} file(s) unchanged. No embeddings created."
        )
        return

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if manifest is None:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = client.create_collection(
            COLLECTION_NAME,
            embedding_function=None,
            metadata={
                "hnsw:space": "cosine",
                "version": STORE_VERSION,
                "model": config.EMBEDDING_MODEL,
                "dim": embeddings.dimension(),
            },
        )
    else:
        collection = client.get_collection(COLLECTION_NAME)

    changed_sources = {entry["source"] for entry in to_add}
    for name in set(removed) | changed_sources:
        collection.delete(where={"source": name})  # clears stale/removed chunks; no-op for brand-new files

    embed_entries(to_add)

    if to_add:
        collection.add(
            ids=[e["id"] for e in to_add],
            embeddings=[e["embedding"] for e in to_add],
            documents=[e["text"] for e in to_add],
            metadatas=[{"source": e["source"]} for e in to_add],
        )

    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(
        json.dumps(
            {
                "version": STORE_VERSION,
                "model": config.EMBEDDING_MODEL,
                "chunkChars": config.MAX_CHUNK_CHARS,
                "chunkOverlap": config.MAX_CHUNK_OVERLAP,
                "files": file_hashes,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(
        f"Done. {len(files)} files: {changed_files} re-embedded, {reused_files} reused, "
        f"{len(removed)} removed -> {collection.count()} chunks."
    )


if __name__ == "__main__":
    main()


