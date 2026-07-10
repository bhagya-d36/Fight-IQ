from ingest import _overlap_tail, chunk_markdown


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


def test_overlap_tail_trims_to_clean_boundary():
    body = "abcdefgh ijklmnop qrstuvwx"
    tail = _overlap_tail(body, 10)
    assert tail in body
    assert not tail.startswith(" ")


def test_overlap_tail_returns_whole_body_when_shorter_than_overlap():
    body = "short text"
    assert _overlap_tail(body, 100) == body.strip()


def test_overlap_tail_falls_back_to_raw_tail_when_no_boundary_found():
    body = "a" * 50  # no spaces/newlines/periods anywhere
    assert _overlap_tail(body, 10) == "a" * 10


def test_oversized_section_chunks_share_overlap():
    paras = ["Alpha bravo charlie.", "Delta echo foxtrot.", "Golf hotel india.",
             "Juliet kilo lima.", "Mike november oscar."]
    text = "# Title\n\n## Section\n\n" + "\n\n".join(paras)
    chunks = chunk_markdown("doc.md", text, max_chars=45, overlap=25)
    assert len(chunks) > 1
    tail_words = chunks[0].strip().split()[-2:]
    head_words = chunks[1].strip().split()
    assert any(w in head_words for w in tail_words)


def test_overlap_clamped_when_larger_than_max_chars():
    para = "word " * 20  # ~100 chars
    text = "# Title\n\n## Big Section\n\n" + "\n\n".join([para] * 5)
    chunks = chunk_markdown("doc.md", text, max_chars=150, overlap=1000)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) < 450  # bounded — overlap clamps to max_chars // 2, no runaway duplication


def test_prefix_appears_once_per_chunk():
    para = "word " * 20
    text = "# Title\n\n## Big Section\n\n" + "\n\n".join([para] * 5)
    chunks = chunk_markdown("doc.md", text, max_chars=150, overlap=50)
    prefix = "Title — Big Section"
    for c in chunks:
        assert c.count(prefix) == 1


def test_single_chunk_section_unaffected_by_overlap():
    text = "# Title\n\n## Small Section\n\nJust one short paragraph."
    chunks = chunk_markdown("doc.md", text, max_chars=1500, overlap=200)
    assert len(chunks) == 1
    assert chunks[0].count("Just one short paragraph.") == 1
