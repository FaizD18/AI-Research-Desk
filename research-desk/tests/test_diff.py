"""Tests for year-over-year risk diffing.

The classification logic is pure and is tested with hand-constructed unit
embeddings — no model download. A ``slow`` test exercises the real embedding
model on the actual extracted AAPL filings.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_desk import config, diff
from research_desk.diff import RiskChange, classify_changes, load_paragraphs


def _counts(changes: list[RiskChange]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in changes:
        out[c.change_type] = out.get(c.change_type, 0) + 1
    return out


# --- load_paragraphs -----------------------------------------------------


def test_load_paragraphs_drops_heading_and_short_blocks() -> None:
    long = " ".join(["word"] * 40)
    text = f"Item 1A. Risk Factors\n\nCredit risk\n\n{long}\n\n{long}"
    paras = load_paragraphs(text)
    assert paras == [long, long]  # heading + short subheading dropped


# --- classify_changes ----------------------------------------------------


def test_classify_bands() -> None:
    """One UNCHANGED, one ESCALATED, one NEW (current) and one REMOVED (prior)."""
    prior_paras = ["supply chain risk", "currency risk"]
    curr_paras = ["supply chain risk", "supply chain risk expanded", "brand new risk"]
    # 3-D unit vectors chosen to hit each cosine band precisely.
    prior_emb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    curr_emb = np.array(
        [
            [1.0, 0.0, 0.0],   # cos 1.00 with prior[0]     -> UNCHANGED
            [0.85, 0.0, 0.53],  # cos 0.85 with prior[0]     -> ESCALATED
            [0.0, 0.0, 1.0],   # cos 0.00 with both priors  -> NEW
        ]
    )
    changes = classify_changes(prior_paras, curr_paras, prior_emb, curr_emb)
    assert _counts(changes) == {"UNCHANGED": 1, "ESCALATED": 1, "NEW": 1, "REMOVED": 1}

    unchanged = next(c for c in changes if c.change_type == "UNCHANGED")
    assert unchanged.similarity == pytest.approx(1.0, abs=1e-6)
    assert unchanged.matched_text == "supply chain risk"

    escalated = next(c for c in changes if c.change_type == "ESCALATED")
    assert config.MATCH_THRESHOLD <= escalated.similarity < config.UNCHANGED_THRESHOLD
    assert escalated.matched_text == "supply chain risk"

    removed = next(c for c in changes if c.change_type == "REMOVED")
    assert removed.paragraph_text == "currency risk"
    assert removed.matched_text is None


def test_first_year_all_new() -> None:
    curr = ["a risk", "another risk"]
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    changes = classify_changes([], curr, np.zeros((0, 0)), emb)
    assert _counts(changes) == {"NEW": 2}
    assert all(c.matched_text is None and c.similarity is None for c in changes)


def test_all_removed_when_current_empty() -> None:
    prior = ["a risk", "another risk"]
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    changes = classify_changes(prior, [], emb, np.zeros((0, 0)))
    assert _counts(changes) == {"REMOVED": 2}


def test_removed_only_when_below_threshold() -> None:
    """A prior paragraph with a strong current match is not REMOVED."""
    prior = ["retained risk"]
    curr = ["retained risk reworded"]
    prior_emb = np.array([[1.0, 0.0]])
    curr_emb = np.array([[0.8, 0.6]])  # cos 0.8 -> ESCALATED, prior kept
    changes = classify_changes(prior, curr, prior_emb, curr_emb)
    assert _counts(changes) == {"ESCALATED": 1}


def test_thresholds_are_honored() -> None:
    prior = ["r"]
    curr = ["r"]
    emb_hi = np.array([[1.0, 0.0]])
    emb_mid = np.array([[0.8, 0.6]])  # cos 0.8
    assert classify_changes(prior, curr, emb_hi, emb_hi)[0].change_type == "UNCHANGED"
    assert classify_changes(prior, curr, emb_hi, emb_mid)[0].change_type == "ESCALATED"
    emb_lo = np.array([[0.5, 0.87]])  # cos ~0.5 < MATCH_THRESHOLD
    assert classify_changes(prior, curr, emb_hi, emb_lo)[0].change_type == "NEW"


# --- Real embedding integration (slow; downloads the model) --------------


@pytest.mark.slow
def test_real_aapl_diff_is_sensible() -> None:
    """Diffing real consecutive AAPL filings yields a plausible mix and
    valid similarity scores in [MATCH_THRESHOLD, 1]."""
    import sqlite3
    from pathlib import Path

    if not config.DB_PATH.exists():
        pytest.skip("no ingested/extracted data — run `make ingest && make extract`")
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT f.accession, e.text_path FROM extractions e
           JOIN filings f ON f.accession = e.accession
           WHERE f.ticker='AAPL' ORDER BY f.report_date""",
    ).fetchall()
    conn.close()
    if len(rows) < 2:
        pytest.skip("need at least two extracted AAPL filings")

    prior, curr = rows[-2], rows[-1]
    prior_paras = load_paragraphs(Path(prior["text_path"]).read_text())
    curr_paras = load_paragraphs(Path(curr["text_path"]).read_text())
    changes = classify_changes(
        prior_paras,
        curr_paras,
        diff.embed(prior["accession"], prior_paras),
        diff.embed(curr["accession"], curr_paras),
    )
    counts = _counts(changes)
    # Two consecutive years of the same company: most risks persist.
    assert counts.get("UNCHANGED", 0) + counts.get("ESCALATED", 0) > 0
    for c in changes:
        if c.similarity is not None:
            assert config.MATCH_THRESHOLD <= c.similarity <= 1.0 + 1e-6
