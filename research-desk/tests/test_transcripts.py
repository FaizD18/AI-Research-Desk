"""Tests for transcript ingestion and tone metrics.

No network: fetch is faked or bypassed and metric extraction runs over
committed-style fixture payloads written to a temporary cache directory,
exactly the shape fetch_transcripts() writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_desk import config, db, transcripts

# 10 words; 2 hedging ("we believe" phrase + "may"), 2 uncertainty
# ("risks", "uncertain"), 1 guidance ("guidance").
_SAMPLE = "We believe revenue may grow. Risks remain uncertain. Guidance unchanged."


@pytest.fixture()
def _tmp_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    return tmp_path


def _write_payload(
    ticker: str,
    fy: int,
    fq: int,
    paragraphs: list[dict],
    call_date: str = "2026-01-29",
) -> Path:
    path = config.TRANSCRIPTS_DIR / ticker / f"fy{fy}q{fq}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "ticker": ticker,
                "fiscal_year": fy,
                "fiscal_quarter": fq,
                "call_date": call_date,
                "source": "defeatbeta",
                "paragraphs": paragraphs,
            }
        )
    )
    return path


def _long_paragraphs(n_words: int = 600) -> list[dict]:
    """Management speech long enough to clear TRANSCRIPT_MIN_WORDS."""
    filler = " ".join(["revenue"] * n_words)
    return [
        {"speaker": "Jane CEO", "content": _SAMPLE},
        {"speaker": "Jane CEO", "content": filler},
    ]


# --- tone_metrics ---------------------------------------------------------


def test_tone_metrics_counts_terms_per_1k_words() -> None:
    m = transcripts.tone_metrics(_SAMPLE)
    assert m["word_count"] == 10
    assert m["hedging_per_1k"] == 200.0  # "we believe" (phrase) + "may"
    assert m["uncertainty_per_1k"] == 200.0  # "risks" + "uncertain"
    assert m["guidance_per_1k"] == 100.0  # "guidance"


def test_tone_metrics_is_case_insensitive_and_word_bounded() -> None:
    # "MAYBE" must not match "may"; "RISKS" must match "risks".
    m = transcripts.tone_metrics("MAYBE the RISKS grow")
    assert m["hedging_per_1k"] == 0.0
    assert m["uncertainty_per_1k"] == 250.0  # 1 of 4 words


def test_tone_metrics_empty_text_returns_zero_rates() -> None:
    m = transcripts.tone_metrics("")
    assert m["word_count"] == 0
    assert m["hedging_per_1k"] == 0.0


# --- run_transcripts -------------------------------------------------------


def _run_offline(monkeypatch, tickers: list[str]) -> dict[str, int]:
    """Run the stage with the network fetch unavailable (cache-only path)."""

    def _no_network(ticker: str, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(transcripts, "fetch_transcripts", _no_network)
    return transcripts.run_transcripts(tickers)


def test_run_transcripts_stores_metrics_from_cache(_tmp_dirs, monkeypatch) -> None:
    _write_payload("AAPL", 2026, 1, _long_paragraphs(), call_date="2026-01-29")
    _write_payload("AAPL", 2026, 2, _long_paragraphs(), call_date="2026-04-30")

    assert _run_offline(monkeypatch, ["AAPL"]) == {"AAPL": 2}

    conn = db.connect(config.DB_PATH)
    rows = conn.execute(
        "SELECT * FROM transcripts ORDER BY fiscal_year, fiscal_quarter"
    ).fetchall()
    conn.close()
    assert [(r["fiscal_year"], r["fiscal_quarter"]) for r in rows] == [(2026, 1), (2026, 2)]
    assert rows[0]["call_date"] == "2026-01-29"
    assert rows[0]["word_count"] == 610
    assert rows[0]["hedging_per_1k"] == pytest.approx(2 / 610 * 1000, abs=0.01)


def test_run_transcripts_excludes_operator_speech(_tmp_dirs, monkeypatch) -> None:
    paragraphs = _long_paragraphs() + [
        {"speaker": "Operator", "content": " ".join(["uncertainty"] * 400)},
    ]
    _write_payload("NVDA", 2026, 1, paragraphs)

    _run_offline(monkeypatch, ["NVDA"])

    conn = db.connect(config.DB_PATH)
    row = conn.execute("SELECT * FROM transcripts").fetchone()
    conn.close()
    assert row["word_count"] == 610  # operator's 400 words not counted
    assert row["uncertainty_per_1k"] == pytest.approx(2 / 610 * 1000, abs=0.01)


def test_run_transcripts_skips_stub_and_malformed_files(_tmp_dirs, monkeypatch) -> None:
    _write_payload("JPM", 2025, 1, [{"speaker": "CEO", "content": "too short"}])
    bad = config.TRANSCRIPTS_DIR / "JPM" / "fy2025q2.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{not json")
    _write_payload("JPM", 2025, 3, _long_paragraphs())

    assert _run_offline(monkeypatch, ["JPM"]) == {"JPM": 1}


def test_run_transcripts_upsert_keeps_row_id_stable(_tmp_dirs, monkeypatch) -> None:
    _write_payload("AAPL", 2026, 1, _long_paragraphs())

    _run_offline(monkeypatch, ["AAPL"])
    conn = db.connect(config.DB_PATH)
    first_id = conn.execute("SELECT id FROM transcripts").fetchone()["id"]
    conn.close()

    _run_offline(monkeypatch, ["AAPL"])  # rerun: update, not duplicate
    conn = db.connect(config.DB_PATH)
    rows = conn.execute("SELECT id FROM transcripts").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0]["id"] == first_id  # theses referencing it stay valid
