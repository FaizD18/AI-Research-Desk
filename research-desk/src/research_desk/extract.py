"""Extract Item 1A (Risk Factors) from cached 10-K HTML.

Strategy (verified against real AAPL/NVDA/JPM filings):

1. Parse once with lxml, drop the hidden inline-XBRL ``<ix:header>`` subtree
   (1.2 MB of machine metadata in JPM filings), and flatten the document into
   an ordered list of leaf text blocks with normalized text.
2. PRIMARY locator — ToC anchor walk: modern filings hyperlink the table of
   contents to empty anchor ``<div id=...>`` elements placed immediately
   before each section heading. Resolving the "Item 1A" / "Item 1B" ToC links
   to their targets bounds the section exactly and is immune to the
   cross-reference false positives that break text matching.
3. SECONDARY locator — heading scan: for filings without a linked ToC, find a
   short block whose whole text is the Item 1A heading. Every observed
   cross-reference is a full sentence, so a length cap kills them all. Visual
   cues are deliberately not used: JPM's heading is not bold and NVDA's is
   green.
4. A deterministic validation gate accepts or rejects every candidate slice.
5. LLM fallback ONLY when heuristics fail validation: the model picks two
   indexes from an enumerated outline of candidate headings — it never sees
   or generates section prose, so it cannot inject text. Every fallback is
   logged; a rising fallback rate means the heuristics need work.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import lxml.html

from research_desk import config, db

log = logging.getLogger(__name__)

BLOCK_TAGS = frozenset(
    {"div", "p", "td", "th", "li", "h1", "h2", "h3", "h4", "h5", "h6"}
)

# Whole-block heading matchers, applied to normalized text.
TOC_1A = re.compile(r"^item\s*1a[.:]?(\s*risk\s*factors\.?)?\s*$", re.I)
TOC_1B = re.compile(r"^item\s*1b[.:]?(\s*unresolved\s*staff\s*comments\.?)?\s*$", re.I)
TOC_2 = re.compile(r"^item\s*2[.:]?(\s*propert\w*\.?)?\s*$", re.I)
HEADING_1A = re.compile(r"^item\s*1a[.:]?\s*risk\s*factors\.?\s*$", re.I)
HEADING_1B = re.compile(r"^item\s*1b\b", re.I)
HEADING_2 = re.compile(r"^item\s*2[.:]?\s*propert", re.I)
FIRST_BLOCK_OK = re.compile(r"^item\s*1a", re.I)

# Print artifacts that leak into extracted text (JPM/AAPL running headers,
# page numbers, and "Apple Inc. | 2025 Form 10-K | 23"-style footers).
PAGE_ARTIFACT = re.compile(r"^(part\s+[ivx]+|\d{1,4}|table\s+of\s+contents)$", re.I)
RUNNING_FOOTER = re.compile(r"^.{0,60}form\s+10-k.{0,15}$", re.I)

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―"), "-")


class ExtractionError(Exception):
    """Raised when no locator produces a slice that passes validation."""

    def __init__(self, message: str, metrics: dict | None = None):
        super().__init__(message)
        self.metrics = metrics or {}


def normalize(text: str) -> str:
    """Collapse whitespace and unify NBSP/dash variants for reliable matching."""
    text = text.replace("\xa0", " ").translate(_DASHES)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class Block:
    """One leaf text block, with its document-order position."""

    pos: int
    text: str

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class ParsedDoc:
    """Flattened view of a filing: ordered blocks plus anchor structure."""

    blocks: list[Block] = field(default_factory=list)
    id_pos: dict[str, int] = field(default_factory=dict)
    # (pos, target_id, normalized link text) for every internal <a href="#...">
    anchors: list[tuple[int, str, str]] = field(default_factory=list)

    @property
    def total_words(self) -> int:
        return sum(b.words for b in self.blocks)


@dataclass(frozen=True)
class ExtractionResult:
    """A validated Item 1A section."""

    paragraphs: list[str]
    method: str  # toc-anchor | heading-scan | llm-boundary
    metrics: dict


def parse_blocks(html: bytes) -> ParsedDoc:
    """Parse filing HTML into ordered leaf blocks, ids, and internal anchors."""
    tree = lxml.html.fromstring(html)

    for el in tree.iter():
        if isinstance(el.tag, str) and el.tag.lower() == "ix:header":
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)
            break  # there is at most one hidden iXBRL header

    doc = ParsedDoc()
    block_els: list[tuple[int, lxml.html.HtmlElement]] = []
    for pos, el in enumerate(tree.iter()):
        if not isinstance(el.tag, str):
            continue  # comments, processing instructions
        el_id = el.get("id")
        if el_id:
            doc.id_pos[el_id] = pos
        tag = el.tag.lower()
        if tag == "a":
            href = el.get("href") or ""
            if href.startswith("#"):
                doc.anchors.append((pos, href[1:], normalize(el.text_content())))
        elif tag in BLOCK_TAGS:
            block_els.append((pos, el))

    for pos, el in block_els:
        # A block is a leaf iff nothing block-level nests inside it — parents
        # would otherwise duplicate their children's text.
        if any(True for _ in el.iterdescendants(*BLOCK_TAGS)):
            continue
        text = normalize(el.text_content())
        if text:
            doc.blocks.append(Block(pos, text))
    return doc


def _strip_artifacts(blocks: list[Block]) -> list[Block]:
    """Drop print pagination noise (running headers, page numbers, footers)."""
    return [
        b
        for b in blocks
        if not (PAGE_ARTIFACT.match(b.text) or RUNNING_FOOTER.match(b.text))
    ]


def validate(blocks: list[Block], total_words: int) -> tuple[bool, dict]:
    """Deterministic acceptance gate for a candidate section slice."""
    word_count = sum(b.words for b in blocks)
    n_paragraphs = sum(1 for b in blocks if b.words >= config.MIN_PARAGRAPH_WORDS)
    fraction = word_count / total_words if total_words else 0.0
    checks = {
        "word_count_in_range": config.EXTRACT_MIN_WORDS
        <= word_count
        <= config.EXTRACT_MAX_WORDS,
        "starts_with_item_1a": bool(blocks) and bool(FIRST_BLOCK_OK.match(blocks[0].text)),
        "no_overshoot_into_1b": not any(
            "unresolved staff comments" in b.text.casefold() for b in blocks[1:]
        ),
        "enough_paragraphs": n_paragraphs >= config.EXTRACT_MIN_PARAGRAPHS,
        "sane_fraction_of_doc": 0.01 <= fraction <= 0.40,
    }
    metrics = {
        "word_count": word_count,
        "n_paragraphs": n_paragraphs,
        "fraction_of_doc": round(fraction, 4),
        "checks": checks,
    }
    return all(checks.values()), metrics


def _slice_between(doc: ParsedDoc, start_pos: int, end_pos: int) -> list[Block]:
    """Blocks strictly between two document positions, artifacts stripped."""
    return _strip_artifacts([b for b in doc.blocks if start_pos < b.pos < end_pos])


def _locate_by_toc_anchor(doc: ParsedDoc) -> list[Block] | None:
    """Resolve ToC hyperlinks to their anchor targets to bound the section."""
    starts = [
        doc.id_pos[target]
        for _, target, text in doc.anchors
        if TOC_1A.match(text) and target in doc.id_pos
    ]
    ends = [
        doc.id_pos[target]
        for _, target, text in doc.anchors
        if (TOC_1B.match(text) or TOC_2.match(text)) and target in doc.id_pos
    ]
    for start_pos in starts:
        end_candidates = sorted(pos for pos in ends if pos > start_pos)
        for end_pos in end_candidates:
            section = _slice_between(doc, start_pos, end_pos)
            ok, _ = validate(section, doc.total_words)
            if ok:
                return section
    return None


def _locate_by_heading_scan(doc: ParsedDoc) -> list[Block] | None:
    """Find short whole-block headings; length cap rejects cross-references."""
    blocks = doc.blocks
    for i, block in enumerate(blocks):
        if len(block.text) > config.HEADING_MAX_CHARS or not HEADING_1A.match(block.text):
            continue
        for j in range(i + 1, len(blocks)):
            candidate = blocks[j]
            if len(candidate.text) <= 80 and (
                HEADING_1B.match(candidate.text) or HEADING_2.match(candidate.text)
            ):
                section = _strip_artifacts(blocks[i:j])
                ok, _ = validate(section, doc.total_words)
                if ok:
                    return section
                break  # wrong span for this start; try the next 1A candidate
    return None


def _outline_candidates(doc: ParsedDoc) -> list[tuple[int, Block]]:
    """Candidate heading blocks shown to the LLM fallback (index into blocks)."""
    item_re = re.compile(r"item\s*\d+", re.I)
    return [
        (i, b)
        for i, b in enumerate(doc.blocks)
        if len(b.text) <= 100 and (item_re.search(b.text) or "risk factors" in b.text.casefold())
    ]


def _locate_by_llm(doc: ParsedDoc) -> list[Block] | None:
    """LLM fallback: pick two indexes from an outline — never generate prose."""
    from research_desk import llm  # lazy: heuristics must not require an API key

    candidates = _outline_candidates(doc)
    if len(candidates) < 2:
        return None
    outline = [
        {"index": i, "text": b.text[:80]} for i, b in candidates
    ]
    choice = llm.pick_section_boundaries(outline, section="Item 1A Risk Factors")
    if choice is None:
        return None
    start, end = choice
    if not (0 <= start < end < len(doc.blocks)):
        log.warning("LLM returned out-of-range boundaries (%s, %s)", start, end)
        return None
    section = _strip_artifacts(doc.blocks[start:end])
    ok, _ = validate(section, doc.total_words)
    return section if ok else None


def extract_item_1a(html: bytes, llm_fallback: bool = True) -> ExtractionResult:
    """Extract the validated Item 1A section from one filing's HTML.

    Tries locators in order of trustworthiness; raises ExtractionError with
    the last validation metrics if nothing passes so the caller can
    quarantine the filing rather than feed a bad slice into the diff stage.
    """
    doc = parse_blocks(html)
    for method, locate in (
        ("toc-anchor", _locate_by_toc_anchor),
        ("heading-scan", _locate_by_heading_scan),
    ):
        section = locate(doc)
        if section:
            _, metrics = validate(section, doc.total_words)
            return ExtractionResult([b.text for b in section], method, metrics)

    if llm_fallback:
        log.warning("heuristics failed; falling back to LLM boundary selection")
        section = _locate_by_llm(doc)
        if section:
            _, metrics = validate(section, doc.total_words)
            return ExtractionResult([b.text for b in section], "llm-boundary", metrics)

    # Report the best diagnostic we have: the heading-scan view of the doc.
    _, metrics = validate(_strip_artifacts(doc.blocks), doc.total_words)
    raise ExtractionError("could not locate a valid Item 1A section", metrics)


def run_extract(tickers: list[str]) -> dict[str, int]:
    """Extract Item 1A for every cached filing that isn't already done.

    One bad filing logs and is quarantined (no extractions row); the run
    never crashes. Returns {ticker: sections_extracted}.
    """
    conn = db.connect()
    config.EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}
    fallbacks = 0
    for ticker in tickers:
        rows = conn.execute(
            """SELECT f.accession, f.local_path FROM filings f
               LEFT JOIN extractions e ON e.accession = f.accession
                   AND e.extractor_version = ?
               WHERE f.ticker = ? AND e.accession IS NULL
               ORDER BY f.report_date""",
            (config.EXTRACTOR_VERSION, ticker.upper()),
        ).fetchall()
        done = 0
        for row in rows:
            try:
                html = Path(row["local_path"]).read_bytes()
                result = extract_item_1a(html)
            except Exception:
                log.exception("quarantined %s %s: extraction failed", ticker, row["accession"])
                continue
            if result.method == "llm-boundary":
                fallbacks += 1
            text_path = config.EXTRACTED_DIR / f"{row['accession']}.txt"
            text_path.write_text("\n\n".join(result.paragraphs))
            conn.execute(
                """INSERT OR REPLACE INTO extractions
                   (accession, method, word_count, n_paragraphs, text_path,
                    extractor_version, validation_json, extracted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["accession"],
                    result.method,
                    result.metrics["word_count"],
                    result.metrics["n_paragraphs"],
                    str(text_path),
                    config.EXTRACTOR_VERSION,
                    json.dumps(result.metrics),
                    datetime.now(UTC).isoformat(),
                ),
            )
            conn.commit()
            log.info(
                "%s %s: %d words via %s",
                ticker, row["accession"], result.metrics["word_count"], result.method,
            )
            done += 1
        summary[ticker] = done
    if fallbacks:
        log.warning("LLM fallback used for %d filings — heuristics may need work", fallbacks)
    conn.close()
    return summary
