"""Tests for Item 1A extraction.

Two tiers:

* Fast unit tests over synthetic fixtures that reproduce the exact structural
  patterns found in real AAPL/NVDA/JPM filings (ToC anchor divs, per-filer
  heading markup, cross-reference false positives, print artifacts, hidden
  inline-XBRL headers). These are committed and always run.
* ``slow`` integration tests over the real cached filings in ``data/raw/``
  (present after ``make ingest``). These are the real proof the heuristics
  work; they are excluded from the default ``make test`` run.
"""

from __future__ import annotations

import pytest

from research_desk import config, extract
from research_desk.extract import Block, ExtractionError, normalize, parse_blocks, validate

# --- Synthetic filing builder -------------------------------------------

_RISK_SENTENCE = (
    "The Company faces material risks related to global economic conditions, "
    "competitive pressures, regulatory developments, supply chain disruptions, "
    "and technological change that could adversely affect its business, results "
    "of operations, financial condition, and the trading price of its securities "
    "in ways that are difficult to predict and largely beyond its control."
)


def _para(prefix: str, i: int) -> str:
    return f"<p>{prefix} number {i}: {_RISK_SENTENCE}</p>"


# Per-filer heading markup, matching the real documents byte-for-byte in the
# ways that matter (nbsp entities, colour, weight, trailing punctuation).
HEADINGS = {
    "aapl": (
        '<div style="padding-left:45pt"><span style="font-family:\'Helvetica\';'
        'font-size:9pt;font-weight:700">Item 1A.&#160;&#160;&#160;&#160;Risk Factors</span></div>',
        '<div><span style="font-size:9pt;font-weight:700">'
        "Item 1B.&#160;&#160;&#160;&#160;Unresolved Staff Comments</span></div>",
    ),
    "nvda": (
        '<div><span style="color:#76b900;font-family:\'NVIDIA Sans\';'
        'font-size:10pt;font-weight:700">Item 1A. Risk Factors</span></div>',
        '<div><span style="color:#76b900;font-size:10pt;font-weight:700">'
        "Item 1B. Unresolved Staff Comments</span></div>",
    ),
    # JPM: NOT bold (weight 400), 12pt, trailing period + space.
    "jpm": (
        '<div><span style="color:#000000;font-family:\'Sons\';'
        'font-size:12pt;font-weight:400">Item 1A. Risk Factors. </span></div>',
        '<div><span style="font-size:12pt;font-weight:400">'
        "Item 1B. Unresolved Staff Comments. </span></div>",
    ),
}

CROSS_REFS = {
    "aapl": [
        "The Company is subject to various risks. See Part I, Item 1A of this "
        'Form 10-K under the heading "Risk Factors" for a discussion of these risks.'
    ],
    # NVDA's real filing has cross-references BEFORE the real heading — the
    # case that breaks naive occurrence-counting.
    "nvda": [
        "Our business is subject to numerous risks. Refer to Item 1A. Risk "
        "Factors, including the section titled Risks Related to Regulatory matters, "
        "for additional detail about factors that could affect our results.",
        "For a discussion of competitive dynamics, see Item 1A. Risk Factors below.",
    ],
    # JPM cross-references use a page range and a colon variant.
    "jpm": [
        "For a description of the risks affecting the Firm, refer to Part I, "
        "Item 1A: Risk Factors on pages 9-31 of this Form 10-K."
    ],
}


def build_filing(
    filer: str,
    *,
    linked_toc: bool = True,
    n_risk: int = 40,
    n_pad: int = 90,
    page_artifacts: bool = False,
    ix_header: bool = False,
    short_section: bool = False,
) -> bytes:
    """Assemble a synthetic 10-K reproducing ``filer``'s real structure."""
    head_1a, head_1b = HEADINGS[filer]
    parts: list[str] = ["<html><body>"]

    if ix_header:
        parts.append(
            '<div style="display:none"><ix:header>SECRET_XBRL_METADATA '
            "context entity segment " + "x " * 200 + "</ix:header></div>"
        )

    if linked_toc:
        parts.append(
            '<div id="toc"><table>'
            '<tr><td><a href="#a_item1">Item 1.</a></td></tr>'
            '<tr><td><a href="#a_item1a">Item 1A.</a></td></tr>'
            '<tr><td><a href="#a_item1b">Item 1B.</a></td></tr>'
            '<tr><td><a href="#a_item2">Item 2.</a></td></tr>'
            "</table></div>"
        )
    else:
        # Pre-2020 style: a plain-text ToC with no hyperlinks.
        parts.append("<div>Item 1A. Risk Factors ... 9</div>")

    # Item 1 Business (padding + cross-references that must NOT be picked up).
    parts.append('<div id="a_item1"></div>')
    parts.append('<div><span style="font-weight:700">Item 1. Business</span></div>')
    for ref in CROSS_REFS[filer]:
        parts.append(f"<p>{ref}</p>")
    for i in range(n_pad // 2):
        parts.append(_para("Business paragraph", i))

    # Item 1A Risk Factors.
    if linked_toc:
        parts.append('<div id="a_item1a"></div>')
    parts.append(head_1a)
    n = 3 if short_section else n_risk
    for i in range(n):
        parts.append(_para("Risk factor", i))
        if page_artifacts and i % 8 == 7:
            parts.append('<hr style="page-break-after:always"/>')
            parts.append("<div>Part I</div>")
            parts.append(f"<div>{9 + i}</div>")
            parts.append("<div>JPMorgan Chase &amp; Co./2024 Form 10-K</div>")

    # Item 1B (end boundary) + Item 2 padding.
    if linked_toc:
        parts.append('<div id="a_item1b"></div>')
    parts.append(head_1b)
    parts.append("<p>None.</p>")
    if linked_toc:
        parts.append('<div id="a_item2"></div>')
    parts.append('<div><span style="font-weight:700">Item 2. Properties</span></div>')
    for i in range(n_pad // 2):
        parts.append(_para("Properties paragraph", i))

    parts.append("</body></html>")
    return "".join(parts).encode()


# --- normalize / validate unit tests ------------------------------------


def test_normalize_unifies_nbsp_and_dashes() -> None:
    assert normalize("Item\xa01A. Risk—Factors") == "Item 1A. Risk-Factors"
    assert normalize("a–b—c") == "a-b-c"


def test_parse_blocks_drops_ix_header() -> None:
    doc = parse_blocks(build_filing("aapl", ix_header=True))
    joined = " ".join(b.text for b in doc.blocks)
    assert "SECRET_XBRL_METADATA" not in joined


def test_validate_rejects_short_section() -> None:
    blocks = [Block(0, "Item 1A. Risk Factors")] + [
        Block(i, "short") for i in range(1, 5)
    ]
    ok, metrics = validate(blocks, total_words=100_000)
    assert not ok
    assert not metrics["checks"]["word_count_in_range"]


# --- ToC-anchor locator (the primary path) ------------------------------


@pytest.mark.parametrize("filer", ["aapl", "nvda", "jpm"])
def test_toc_anchor_extraction(filer: str) -> None:
    result = extract.extract_item_1a(build_filing(filer), llm_fallback=False)
    assert result.method == "toc-anchor"
    assert result.paragraphs[0].lower().startswith("item 1a")
    assert result.metrics["word_count"] >= config.EXTRACT_MIN_WORDS


@pytest.mark.parametrize("filer", ["aapl", "nvda", "jpm"])
def test_cross_references_excluded(filer: str) -> None:
    """Cross-reference sentences (esp. NVDA's, which precede the heading) and
    the ToC row must not leak into the extracted section."""
    result = extract.extract_item_1a(build_filing(filer), llm_fallback=False)
    body = "\n".join(result.paragraphs).lower()
    assert "business paragraph" not in body  # Item 1 padding stayed out
    assert "refer to item 1a" not in body  # NVDA-style pre-heading cross-ref
    assert "for a discussion" not in body
    # Exactly one line is the heading; no cross-ref duplicated it.
    heading_lines = [p for p in result.paragraphs if p.lower().startswith("item 1a")]
    assert len(heading_lines) == 1


def test_page_artifacts_stripped() -> None:
    result = extract.extract_item_1a(
        build_filing("jpm", page_artifacts=True), llm_fallback=False
    )
    body = "\n".join(result.paragraphs)
    assert "Part I" not in body
    assert "Form 10-K" not in body
    assert not any(p.strip().isdigit() for p in result.paragraphs)


# --- Heading-scan fallback (no linked ToC, e.g. pre-2020) ----------------


@pytest.mark.parametrize("filer", ["aapl", "nvda", "jpm"])
def test_heading_scan_when_no_linked_toc(filer: str) -> None:
    result = extract.extract_item_1a(
        build_filing(filer, linked_toc=False), llm_fallback=False
    )
    assert result.method == "heading-scan"
    assert result.paragraphs[0].lower().startswith("item 1a")


# --- Quarantine on failure ----------------------------------------------


def test_short_section_raises_extraction_error() -> None:
    with pytest.raises(ExtractionError) as exc:
        extract.extract_item_1a(
            build_filing("aapl", short_section=True), llm_fallback=False
        )
    assert "word_count" in exc.value.metrics


def test_run_extract_quarantines_bad_filing(tmp_path, monkeypatch) -> None:
    """A filing that can't be parsed logs and is skipped; the run never raises
    and emits no extractions row."""
    from research_desk import db

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(config, "EXTRACTED_DIR", tmp_path / "extracted")
    conn = db.connect(config.DB_PATH)
    bad = tmp_path / "bad.html"
    bad.write_bytes(b"<html><body><p>no risk factors here at all</p></body></html>")
    conn.execute(
        """INSERT INTO filings (accession, ticker, cik, form, filing_date,
           report_date, primary_document, doc_url, local_path, sha256,
           size_bytes, fetched_at) VALUES
           ('x-1','AAPL',320193,'10-K','2025-10-31','2025-09-27','d.htm',
            'http://x', ?, 'sha', 10, '2025-01-01')""",
        (str(bad),),
    )
    conn.commit()
    conn.close()

    summary = extract.run_extract(["AAPL"])
    assert summary == {"AAPL": 0}
    conn = db.connect(config.DB_PATH)
    assert conn.execute("SELECT COUNT(*) FROM extractions").fetchone()[0] == 0
    conn.close()


# --- Real-filing integration (slow; needs `make ingest`) -----------------


@pytest.mark.slow
def test_real_filings_extract_with_heuristics() -> None:
    """Every cached real filing must extract via a heuristic method (no LLM
    fallback) with a plausible word count."""
    import sqlite3

    if not config.DB_PATH.exists():
        pytest.skip("no ingested data — run `make ingest` first")
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT ticker, accession, local_path FROM filings").fetchall()
    conn.close()
    assert rows, "expected ingested filings"

    from pathlib import Path

    for row in rows:
        html = Path(row["local_path"]).read_bytes()
        result = extract.extract_item_1a(html, llm_fallback=False)
        assert result.method in ("toc-anchor", "heading-scan"), (
            f"{row['ticker']} {row['accession']} needed LLM fallback"
        )
        assert config.EXTRACT_MIN_WORDS <= result.metrics["word_count"] <= config.EXTRACT_MAX_WORDS
        assert result.paragraphs[0].lower().startswith("item 1a")
