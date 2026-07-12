"""Tests for thesis generation: prompt assembly, point-in-time filing
selection, DB wiring, and graceful degradation. The LLM batch is faked —
no key, no network.
"""

from __future__ import annotations

import json

import pytest

from research_desk import config, db, llm, thesis
from research_desk.thesis import EvidenceItem, Thesis


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "th.db")
    connection = db.connect(config.DB_PATH)
    yield connection
    connection.close()


def _seed_filing(conn, accession: str, ticker: str, filing_date: str) -> None:
    conn.execute(
        """INSERT INTO filings (accession, ticker, cik, form, filing_date,
           report_date, primary_document, doc_url, local_path, sha256,
           size_bytes, fetched_at)
           VALUES (?, ?, 1, '10-K', ?, ?, 'd', 'u', 'l', 's', 1, 't')""",
        (accession, ticker, filing_date, filing_date),
    )


def _seed_changes(conn, curr: str, prior: str, ticker: str) -> None:
    rows = [
        ("NEW", "a brand new supply chain risk", None),
        ("ESCALATED", "an escalated regulatory risk", 0.8),
        ("UNCHANGED", "same old risk", 0.99),
    ]
    for i, (ct, text, sim) in enumerate(rows):
        conn.execute(
            """INSERT INTO risk_changes (ticker, prior_accession, curr_accession,
               change_type, similarity, paragraph_text, matched_text, paragraph_index)
               VALUES (?, ?, ?, ?, ?, ?, NULL, ?)""",
            (ticker, prior, curr, ct, sim, text, i),
        )


def _seed_transcript(
    conn, ticker: str, fy: int, fq: int, call_date: str, hedging: float = 5.0
) -> int:
    cur = conn.execute(
        """INSERT INTO transcripts (ticker, fiscal_year, fiscal_quarter, call_date,
           word_count, hedging_per_1k, uncertainty_per_1k, guidance_per_1k,
           text_path, source, ingested_at)
           VALUES (?, ?, ?, ?, 8000, ?, 2.0, 4.0, 'p', 'defeatbeta', 't')""",
        (ticker, fy, fq, call_date, hedging),
    )
    return cur.lastrowid


def _full_seed(conn) -> None:
    """One 10-K comparison filed 2025-11-01 + two later calls."""
    _seed_filing(conn, "prior-1", "AAPL", "2024-11-01")
    _seed_filing(conn, "curr-1", "AAPL", "2025-11-01")
    _seed_changes(conn, "curr-1", "prior-1", "AAPL")
    _seed_transcript(conn, "AAPL", 2026, 1, "2026-01-29", hedging=5.0)
    _seed_transcript(conn, "AAPL", 2026, 2, "2026-04-30", hedging=7.5)
    conn.commit()


# --- build_prompt ----------------------------------------------------------


def test_build_prompt_contains_counts_excerpts_and_qoq_delta(conn) -> None:
    _full_seed(conn)
    row = conn.execute(
        "SELECT * FROM transcripts WHERE fiscal_quarter = 2"
    ).fetchone()
    prompt = thesis.build_prompt(conn, row)
    assert prompt is not None
    assert "1 NEW, 1 ESCALATED, 0 REMOVED, 1 UNCHANGED" in prompt
    assert "brand new supply chain risk" in prompt
    assert "unscored" in prompt  # no risk_scores seeded
    assert "hedging +2.5" in prompt  # 7.5 - 5.0 QoQ delta
    assert 'SOURCE "10-K curr-1"' in prompt
    assert "As of: 2026-04-30" in prompt


def test_build_prompt_first_quarter_has_no_prior(conn) -> None:
    _full_seed(conn)
    row = conn.execute(
        "SELECT * FROM transcripts WHERE fiscal_quarter = 1"
    ).fetchone()
    prompt = thesis.build_prompt(conn, row)
    assert "prior quarter: none available" in prompt


def test_build_prompt_returns_none_before_any_filing(conn) -> None:
    """A call that predates every 10-K comparison has no risk signal."""
    _seed_filing(conn, "prior-1", "NVDA", "2025-02-01")
    _seed_filing(conn, "curr-1", "NVDA", "2026-02-01")
    _seed_changes(conn, "curr-1", "prior-1", "NVDA")
    tid = _seed_transcript(conn, "NVDA", 2025, 3, "2025-05-20")
    conn.commit()
    row = conn.execute("SELECT * FROM transcripts WHERE id = ?", (tid,)).fetchone()
    assert thesis.build_prompt(conn, row) is None


def test_build_prompt_picks_latest_filing_on_or_before_call(conn) -> None:
    """Two comparisons exist; the call must use the newer one filed before it."""
    for acc, fdate in (("old-p", "2023-11-01"), ("old-c", "2024-11-01"),
                       ("new-c", "2025-11-01")):
        _seed_filing(conn, acc, "AAPL", fdate)
    _seed_changes(conn, "old-c", "old-p", "AAPL")
    _seed_changes(conn, "new-c", "old-c", "AAPL")
    tid = _seed_transcript(conn, "AAPL", 2025, 2, "2025-05-01")  # before new-c
    conn.commit()
    row = conn.execute("SELECT * FROM transcripts WHERE id = ?", (tid,)).fetchone()
    prompt = thesis.build_prompt(conn, row)
    assert 'SOURCE "10-K old-c"' in prompt  # not new-c: filed after the call


# --- run_thesis --------------------------------------------------------------


def _fake_thesis() -> Thesis:
    return Thesis(
        direction="long",
        confidence=0.7,
        summary="Risk language stable while guidance tone improved.",
        evidence=[EvidenceItem(source="10-K curr-1", point="only one new risk")],
    )


def test_run_thesis_stores_rows_with_evidence_json(conn, monkeypatch) -> None:
    _full_seed(conn)
    monkeypatch.setattr(
        llm, "cached_batch", lambda s, users, m, **k: [_fake_thesis() for _ in users]
    )
    assert thesis.run_thesis(["AAPL"]) == {"AAPL": 2}

    check = db.connect(config.DB_PATH)
    rows = check.execute("SELECT * FROM theses ORDER BY as_of").fetchall()
    check.close()
    assert [r["as_of"] for r in rows] == ["2026-01-29", "2026-04-30"]
    assert rows[0]["direction"] == "long"
    assert rows[0]["confidence"] == 0.7
    evidence = json.loads(rows[0]["evidence_json"])
    assert evidence[0]["source"] == "10-K curr-1"


def test_run_thesis_is_idempotent(conn, monkeypatch) -> None:
    _full_seed(conn)
    monkeypatch.setattr(
        llm, "cached_batch", lambda s, users, m, **k: [_fake_thesis() for _ in users]
    )
    assert thesis.run_thesis(["AAPL"]) == {"AAPL": 2}
    assert thesis.run_thesis(["AAPL"]) == {"AAPL": 0}  # nothing left to do


def test_run_thesis_degrades_gracefully_without_key(conn, monkeypatch) -> None:
    _full_seed(conn)
    monkeypatch.setattr(llm, "cached_batch", lambda s, users, m, **k: [None] * len(users))
    assert thesis.run_thesis(["AAPL"]) == {"AAPL": 0}
    check = db.connect(config.DB_PATH)
    assert check.execute("SELECT COUNT(*) FROM theses").fetchone()[0] == 0
    check.close()


def test_run_thesis_rejects_invalid_direction() -> None:
    with pytest.raises(Exception):
        Thesis(direction="hold", confidence=0.5, summary="s", evidence=[])
