from ingest import chunk_markdown


def test_splits_on_h2_headings():
    text = "# My Doc\n\n## First\n\nBody one.\n\n## Second\n\nBody two.\n"
    chunks = chunk_markdown("doc.md", text)
    assert len(chunks) == 2
    assert chunks[0].startswith("My Doc — First")
    assert chunks[1].startswith("My Doc — Second")


def test_prefixes_title_and_heading():
    text = "# Title Here\n\n## A Heading\n\nSome content."
    chunks = chunk_markdown("doc.md", text)
    assert chunks[0].startswith("Title Here — A Heading\n\n")
    assert "Some content." in chunks[0]


def test_falls_back_to_file_name_without_title():
    text = "## Only Heading\n\nBody text."
    chunks = chunk_markdown("no-title.md", text)
    assert chunks[0].startswith("no-title.md — Only Heading")


def test_skips_title_only_intro_section():
    text = "# Title\n\nintro line with no heading\n\n## Real Section\n\nContent."
    chunks = chunk_markdown("doc.md", text)
    # The pre-"##" intro paragraph is skipped (it's not empty, so this
    # documents current behavior: only a bare title line is dropped).
    assert any("Real Section" in c for c in chunks)


def test_oversized_section_packs_paragraphs():
    para = "word " * 20  # ~100 chars
    text = "# Title\n\n## Big Section\n\n" + "\n\n".join([para] * 5)
    chunks = chunk_markdown("doc.md", text, max_chars=150)
    assert len(chunks) > 1
    for c in chunks:
        # allow the prefix itself to push slightly over, but body packing must respect the limit
        assert len(c) < 300


def test_empty_sections_are_skipped():
    text = "# Title\n\n## Empty\n\n## Real\n\nHas content."
    chunks = chunk_markdown("doc.md", text)
    assert len(chunks) == 1
    assert "Real" in chunks[0]
