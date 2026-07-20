"""fetch_data.py — pulls UFC content from Wikipedia's public API and writes it
as Markdown into knowledge-base/wikipedia/, shaped so ingest.py can chunk it
the same way it chunks the hand-authored files (one "# Title", one "##
Section" per topic).

Uses only the stdlib (urllib) against the free MediaWiki API — no API key,
no extra dependencies. Re-run any time; it overwrites files in place and
ingest.py's content hashing means only actually-changed pages get re-embedded.

Fetch everything (events + roster + curated pages):  python fetch_data.py
Fetch fighters + curated pages, skip events:          python fetch_data.py --fighters-only
Fetch events + curated pages, skip roster:            python fetch_data.py --events-only
Fetch just the curated seed pages (fast, for testing): python fetch_data.py --seed-only
"""

import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import config  # noqa: F401  (import triggers load_dotenv(), read by prefer_ipv4() below)
from net_fix import prefer_ipv4

prefer_ipv4()

# Some fetched titles contain characters outside the Windows console's default
# codepage (e.g. Procházka); without this, printing them crashes the run.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "knowledge-base" / "wikipedia"

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "FightIQ-KB-Builder/1.0 (personal RAG project; contact: n/a)"
REQUEST_DELAY_SECONDS = 0.8
MAX_RETRIES = 5

SEED_ONLY = "--seed-only" in sys.argv
FIGHTERS_ONLY = "--fighters-only" in sys.argv
EVENTS_ONLY = "--events-only" in sys.argv

SEED_TITLES = [
    "Ultimate Fighting Championship",
    "History of the Ultimate Fighting Championship",
    "UFC rankings",
    "List of UFC champions",
    "List of UFC records",
    "List of UFC bonus award recipients",
    "The Ultimate Fighter",
    "Unified Rules of Mixed Martial Arts",
    "List of UFC events",
    "UFC Hall of Fame",
    "Octagon (mixed martial arts)",
]

# There's no single flat Wikipedia category covering every UFC event (they're
# split across per-year/per-venue categories instead), so events are found by
# title prefix — Wikipedia's naming convention puts every event under "UFC ...".
EVENT_TITLE_PREFIX = "UFC"
ROSTER_PAGE = "List of current UFC fighters"

# "UFC"-prefixed pages that aren't event articles: unrelated topics that
# happen to share the prefix (a gene, a French consumer magazine, ...),
# meta/nav pages, and non-event media.
EVENT_TITLE_BLOCKLIST = re.compile(
    r"\((gene|disambiguation|video game|TV series|band|film|magazine|brand)\)|"
    r"\.com$|Que Choisir|Undisputed|UFC Hall of Fame|UFC rankings|Catsup|"
    r"Personal Trainer|^UFC test$|^UFC[A-Z]|"  # unrelated orgs sharing the prefix: UFCW, UFCU, ...
    r"^UFC$|^UFC1$|^UFC5$|^List of|^UFC Awards|UFC Unleashed|UFC Ultimate Insider",
    re.IGNORECASE,
)

# Links from the roster list page that aren't fighters (weight classes,
# meta pages, navigation). Filtered out by title match, case-insensitive.
ROSTER_LINK_BLOCKLIST = re.compile(
    r"^(UFC|Ultimate Fighting Championship|List of|Mixed martial arts|"
    r"Unified Rules|Octagon|The Ultimate Fighter|Weight class|"
    r"Featherweight|Lightweight|Welterweight|Middleweight|"
    r"Light heavyweight|Heavyweight|Flyweight|Bantamweight|Strawweight|"
    r"Category:|Template:|Wikipedia:|Help:|Portal:)",
    re.IGNORECASE,
)

TRAILING_SECTION_NAMES = {
    "references",
    "external links",
    "see also",
    "further reading",
    "notes",
    "bibliography",
    "citations",
}


def api_get(params: dict) -> dict:
    query = urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT})
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < MAX_RETRIES - 1:
                wait = float(exc.headers.get("Retry-After", 5 * (attempt + 1)))
                print(f"  rate limited, waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("unreachable")  # loop always returns or raises


def fetch_allpages_prefix(prefix: str) -> list[str]:
    """All page titles (namespace 0) starting with `prefix`, paginated."""
    titles: list[str] = []
    apcontinue = None
    while True:
        params = {
            "action": "query",
            "list": "allpages",
            "apprefix": prefix,
            "apnamespace": "0",
            "aplimit": "max",
        }
        if apcontinue:
            params["apcontinue"] = apcontinue
        data = api_get(params)
        titles.extend(p["title"] for p in data.get("query", {}).get("allpages", []))
        apcontinue = data.get("continue", {}).get("apcontinue")
        time.sleep(REQUEST_DELAY_SECONDS)
        if not apcontinue:
            break
    return titles


def fetch_page_links(title: str) -> list[str]:
    """All namespace-0 links out of a single page, paginated."""
    titles: list[str] = []
    plcontinue = None
    while True:
        params = {
            "action": "query",
            "prop": "links",
            "titles": title,
            "plnamespace": "0",
            "pllimit": "max",
        }
        if plcontinue:
            params["plcontinue"] = plcontinue
        data = api_get(params)
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            titles.extend(link["title"] for link in page.get("links", []))
        plcontinue = data.get("continue", {}).get("plcontinue")
        time.sleep(REQUEST_DELAY_SECONDS)
        if not plcontinue:
            break
    return titles


def fetch_extract(title: str) -> tuple[str, str] | None:
    """Plain-text extract of a single page, following redirects. Returns
    (resolved_title, extract) or None if the page has no content."""
    data = api_get(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "titles": title,
            "redirects": "1",
        }
    )
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if "missing" in page:
            return None
        extract = page.get("extract", "")
        if not extract.strip():
            return None
        return page.get("title", title), extract
    return None


# Wikitable classes that hold navigation/metadata rather than article data
# (page-bottom navboxes, the infobox at the top, cleanup banners) — skipped
# so they don't get attached to whatever heading happens to precede them.
NON_DATA_TABLE_CLASSES = ("navbox", "infobox", "metadata", "ambox", "sidebar")


class _WikiTableParser(HTMLParser):
    """Walks MediaWiki's rendered HTML for a page and records every data
    wikitable, keyed by the heading (h1-h6) it appears under. Only top-level
    tables are captured — tables nested inside another table's cell (common
    in infoboxes) are ignored so they don't inject noise.
    """

    HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables_by_heading: dict[str, list[list[list[tuple[str, int, int]]]]] = {}
        self._current_heading = ""
        self._in_heading = False
        self._heading_buf: list[str] = []

        self._table_depth = 0
        self._table_class = ""
        self._raw_rows: list[list[tuple[str, int, int]]] = []
        self._current_row: list[tuple[str, int, int]] | None = None
        self._in_cell = False
        self._cell_buf: list[str] = []
        self._cell_colspan = 1
        self._cell_rowspan = 1
        self._skip_depth = 0  # inside <sup class="reference">, <style>, <script>

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        if tag in self.HEADING_TAGS:
            self._in_heading = True
            self._heading_buf = []
        elif tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._raw_rows = []
                self._table_class = attrs_d.get("class", "")
        elif tag == "tr" and self._table_depth == 1:
            self._current_row = []
        elif tag in ("td", "th") and self._table_depth == 1:
            self._in_cell = True
            self._cell_buf = []
            try:
                self._cell_colspan = int(attrs_d.get("colspan", 1) or 1)
            except ValueError:
                self._cell_colspan = 1
            try:
                self._cell_rowspan = int(attrs_d.get("rowspan", 1) or 1)
            except ValueError:
                self._cell_rowspan = 1
        elif tag in ("sup", "style", "script") and self._in_cell:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.HEADING_TAGS and self._in_heading:
            self._in_heading = False
            self._current_heading = "".join(self._heading_buf).strip()
        elif tag == "table":
            if self._table_depth == 1 and self._raw_rows:
                is_data_table = not any(c in self._table_class for c in NON_DATA_TABLE_CLASSES)
                if is_data_table and self._current_heading:
                    self.tables_by_heading.setdefault(self._current_heading, []).append(self._raw_rows)
            self._table_depth = max(0, self._table_depth - 1)
        elif tag == "tr" and self._table_depth == 1:
            if self._current_row is not None:
                self._raw_rows.append(self._current_row)
            self._current_row = None
        elif tag in ("td", "th") and self._table_depth == 1 and self._in_cell:
            text = re.sub(r"\s+", " ", "".join(self._cell_buf)).strip()
            if self._current_row is not None:
                self._current_row.append((text, self._cell_colspan, self._cell_rowspan))
            self._in_cell = False
        elif tag in ("sup", "style", "script") and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        if self._in_heading:
            self._heading_buf.append(data)
        elif self._in_cell:
            self._cell_buf.append(data)


def _expand_table(raw_rows: list[list[tuple[str, int, int]]]) -> list[list[str]]:
    """Resolve rowspan/colspan into a fully aligned grid, carrying a spanning
    cell's text down/across into the columns it covers."""
    pending: dict[int, tuple[str, int]] = {}
    out: list[list[str]] = []
    for raw_row in raw_rows:
        row: list[str] = []
        col = 0
        ci = 0
        while ci < len(raw_row) or col in pending:
            if col in pending:
                text, remaining = pending[col]
                row.append(text)
                pending[col] = (text, remaining - 1)
                if pending[col][1] <= 0:
                    del pending[col]
                col += 1
                continue
            text, colspan, rowspan = raw_row[ci]
            ci += 1
            # A colspan cell has one text value for the whole span (there's no
            # native colspan in a markdown table); put it in the first column
            # of the span and leave the rest blank rather than repeating it.
            for i in range(max(colspan, 1)):
                cell_text = text if i == 0 else ""
                row.append(cell_text)
                if rowspan > 1:
                    pending[col + i] = (cell_text, rowspan - 1)
            col += max(colspan, 1)
        out.append(row)
    return out


def rows_to_markdown_table(rows: list[list[str]]) -> str:
    """Render an aligned grid of cell strings as a GFM markdown table."""
    rows = [r for r in rows if any(c.strip() for c in r)]
    if len(rows) < 2:  # header-only or empty tables aren't useful data
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    def esc(cell: str) -> str:
        return cell.replace("|", "\\|").replace("\n", " ").strip()

    header, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(esc(c) for c in header) + " |", "|" + "|".join(["---"] * width) + "|"]
    lines.extend("| " + " | ".join(esc(c) for c in r) + " |" for r in body)
    return "\n".join(lines)


def fetch_tables(title: str) -> dict[str, list[list[list[str]]]]:
    """Every data wikitable on a page, keyed by the heading text it appears
    under, with rowspan/colspan already resolved. Wikipedia's plain-text
    extract API (used by fetch_extract) silently drops all tables, so this
    is the only way to recover records/champions/results that are stored in
    table form rather than prose.
    """
    try:
        data = api_get(
            {
                "action": "parse",
                "page": title,
                "prop": "text",
                "formatversion": "2",
                "redirects": "1",
            }
        )
    except urllib.error.HTTPError:
        return {}
    html = data.get("parse", {}).get("text", "")
    if not html:
        return {}
    parser = _WikiTableParser()
    parser.feed(html)
    return {
        heading: [_expand_table(t) for t in tables]
        for heading, tables in parser.tables_by_heading.items()
    }


def clean_to_markdown(
    title: str, extract: str, tables_by_heading: dict[str, list[list[list[str]]]] | None = None
) -> str | None:
    """Convert a Wikipedia plain-text extract into knowledge-base Markdown.

    "== Section ==" -> "## Section", "=== Sub ===" -> "### Sub", boilerplate
    trailing sections (References, See also, ...) dropped. Any wikitable
    (from `tables_by_heading`) matching a heading is rendered as a markdown
    table right after that heading.
    """
    tables_by_heading = tables_by_heading or {}
    lines = extract.splitlines()
    out: list[str] = [f"# {title}", ""]
    skip_section = False
    body_has_content = False

    for line in lines:
        heading_match = re.match(r"^(={2,6})\s*(.+?)\s*\1$", line.strip())
        if heading_match:
            level = len(heading_match.group(1))
            name = heading_match.group(2).strip()
            skip_section = name.strip().lower() in TRAILING_SECTION_NAMES
            if skip_section:
                continue
            md_level = min(level, 6)
            out.append("")
            out.append(f"{'#' * md_level} {name}")
            out.append("")
            for table_rows in tables_by_heading.get(name, []):
                md_table = rows_to_markdown_table(table_rows)
                if md_table:
                    out.append(md_table)
                    out.append("")
                    body_has_content = True
            continue
        if skip_section:
            continue
        stripped = line.strip()
        if stripped:
            body_has_content = True
        out.append(line)

    if not body_has_content:
        return None

    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    return text


def slugify(title: str) -> str:
    # NFKD + ascii-drop transliterates accented letters to their plain form
    # (Yañez -> Yanez) instead of just deleting them (-> Ya-ez).
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    slug = ascii_title.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-") or "page"


def write_page(title: str) -> bool:
    """Fetch + clean + write one page. Returns True if a file was written."""
    result = fetch_extract(title)
    if result is None:
        print(f"  skip (no content): {title}")
        return False
    resolved_title, extract = result
    tables_by_heading = fetch_tables(resolved_title)
    markdown = clean_to_markdown(resolved_title, extract, tables_by_heading)
    if markdown is None:
        print(f"  skip (empty body): {title}")
        return False
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Resolved title (redirects follow to their target) keeps filenames
    # canonical, so two titles that redirect to the same page collapse into
    # one file instead of duplicating its content.
    path = OUT_DIR / f"{slugify(resolved_title)}.md"
    path.write_text(markdown, encoding="utf-8")
    return True


def _dedupe_short_redirects(titles: list[str]) -> list[str]:
    """Drop a bare event title (e.g. "UFC 300") when a long-form title with
    a fight-card subtitle also exists (e.g. "UFC 300: Pereira vs. Hill") —
    the bare form is virtually always just a redirect to the long form, so
    fetching both wastes an API call pair per event.
    """
    long_prefixes = {t.split(":", 1)[0].strip() for t in titles if ":" in t}
    return [t for t in titles if ":" in t or t not in long_prefixes]


def collect_targets() -> list[str]:
    targets: dict[str, None] = {}  # ordered dedup

    for t in SEED_TITLES:
        targets[t] = None

    if not SEED_ONLY:
        if not FIGHTERS_ONLY:
            print("Fetching UFC event pages (title prefix search)...")
            candidates = fetch_allpages_prefix(EVENT_TITLE_PREFIX)
            events = [t for t in candidates if not EVENT_TITLE_BLOCKLIST.search(t)]
            events = _dedupe_short_redirects(events)
            print(f"  found {len(candidates)} candidates, {len(events)} after filtering/dedup")
            for t in events:
                targets[t] = None

        if not EVENTS_ONLY:
            print("Fetching current roster links...")
            links = fetch_page_links(ROSTER_PAGE)
            roster = [t for t in links if not ROSTER_LINK_BLOCKLIST.match(t)]
            print(f"  found {len(links)} links, {len(roster)} after filtering")
            for t in roster:
                targets[t] = None

    return list(targets.keys())


def main() -> None:
    targets = collect_targets()
    print(f"\nFetching {len(targets)} page(s) from Wikipedia...")

    written = 0
    for i, title in enumerate(targets, start=1):
        try:
            if write_page(title):
                written += 1
        except Exception as exc:
            print(f"  error on {title!r}: {exc}")
        time.sleep(REQUEST_DELAY_SECONDS)
        if i % 50 == 0:
            print(f"  {i}/{len(targets)}...")

    print(f"\nDone. {written}/{len(targets)} page(s) written to {OUT_DIR}")


if __name__ == "__main__":
    main()
