"""Smoke tests for the Streamlit dashboard via streamlit.testing.AppTest.

The app must render without exceptions both on an empty database (fresh
clone: every tab shows its 'run make <target>' hint) and on a seeded one.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from research_desk import config, db

_APP = str(config.PROJECT_ROOT / "src" / "research_desk" / "app.py")


@pytest.fixture()
def _tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "app.db")
    monkeypatch.setattr(config, "BACKTEST_REPORT_PATH", tmp_path / "report.md")
    return tmp_path


def test_app_renders_on_empty_database(_tmp_db) -> None:
    at = AppTest.from_file(_APP)
    at.run(timeout=30)
    assert not at.exception
    # Every LLM-gated tab points at its make target instead of crashing.
    hints = " ".join(block.value for block in at.info)
    assert "make transcripts" in hints and "make thesis" in hints


def test_app_renders_with_seeded_data(_tmp_db) -> None:
    conn = db.connect(config.DB_PATH)
    conn.executemany(
        """INSERT INTO filings (accession, ticker, cik, form, filing_date,
           report_date, primary_document, doc_url, local_path, sha256,
           size_bytes, fetched_at)
           VALUES (?, 'AAPL', 1, '10-K', ?, ?, 'd', 'u', 'l', 's', 1, 't')""",
        [("c0", "2024-11-01", "2024-09-28"), ("c1", "2025-11-01", "2025-09-27")],
    )
    conn.execute(
        """INSERT INTO risk_changes (ticker, prior_accession, curr_accession,
           change_type, similarity, paragraph_text, matched_text, paragraph_index)
           VALUES ('AAPL', 'c0', 'c1', 'NEW', NULL, 'a new risk', NULL, 0)"""
    )
    conn.execute(
        """INSERT INTO transcripts (ticker, fiscal_year, fiscal_quarter, call_date,
           word_count, hedging_per_1k, uncertainty_per_1k, guidance_per_1k,
           text_path, source, ingested_at)
           VALUES ('AAPL', 2026, 1, '2026-01-29', 8000, 5.0, 2.0, 4.0,
                   'p', 'defeatbeta', 't')"""
    )
    conn.commit()
    conn.close()

    at = AppTest.from_file(_APP)
    at.run(timeout=30)
    assert not at.exception
    # Seeded tabs rendered data, not their empty-state hints; unseeded LLM
    # stages still hint. This also proves the DB_PATH monkeypatch applied.
    hints = " ".join(block.value for block in at.info)
    assert "make ingest" not in hints and "make transcripts" not in hints
    assert "make thesis" in hints and "make score" in hints
