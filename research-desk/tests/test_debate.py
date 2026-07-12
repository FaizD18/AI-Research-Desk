"""Tests for the Bull/Bear/Judge debate: stage sequencing, transcript
artifacts, DB wiring, and graceful degradation. All LLM stages are faked.
"""

from __future__ import annotations

import pytest

from research_desk import config, db, debate, llm
from research_desk.debate import Argument, Verdict


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "d.db")
    monkeypatch.setattr(config, "DEBATES_DIR", tmp_path / "debates")
    connection = db.connect(config.DB_PATH)
    yield connection
    connection.close()


def _seed_thesis(conn, ticker: str = "AAPL", as_of: str = "2026-01-29") -> int:
    conn.execute(
        """INSERT INTO transcripts (ticker, fiscal_year, fiscal_quarter, call_date,
           word_count, hedging_per_1k, uncertainty_per_1k, guidance_per_1k,
           text_path, source, ingested_at)
           VALUES (?, 2026, 1, ?, 8000, 5.0, 2.0, 4.0, 'p', 'defeatbeta', 't')""",
        (ticker, as_of),
    )
    tid = conn.execute("SELECT id FROM transcripts WHERE ticker = ?", (ticker,)).fetchone()[0]
    cur = conn.execute(
        """INSERT INTO theses (ticker, transcript_id, as_of, direction, confidence,
           summary, evidence_json, model, created_at)
           VALUES (?, ?, ?, 'long', 0.7, 'a summary',
                   '[{"source": "10-K x", "point": "p"}]', 'm', 't')""",
        (ticker, tid, as_of),
    )
    conn.commit()
    return cur.lastrowid


def _staged_batches(monkeypatch, stages: list[list]):
    """Feed cached_batch one prepared result list per call; capture prompts."""
    calls: list[tuple[str, list[str]]] = []

    def fake(system, users, model_cls, **kwargs):
        calls.append((system, list(users)))
        return stages.pop(0)

    monkeypatch.setattr(llm, "cached_batch", fake)
    return calls


def test_run_debate_stores_verdict_and_transcript(conn, monkeypatch) -> None:
    thesis_id = _seed_thesis(conn)
    calls = _staged_batches(
        monkeypatch,
        [
            [Argument(argument="bull case")],
            [Argument(argument="bear case")],
            [Verdict(conviction=72, reasoning="bull survived")],
        ],
    )

    assert debate.run_debate(["AAPL"]) == {"AAPL": 1}

    check = db.connect(config.DB_PATH)
    row = check.execute("SELECT * FROM debates").fetchone()
    check.close()
    assert row["thesis_id"] == thesis_id
    assert row["conviction"] == 72

    transcript = (config.DEBATES_DIR / "AAPL_2026-01-29.md").read_text()
    assert "## Bull" in transcript and "bull case" in transcript
    assert "## Bear" in transcript and "bear case" in transcript
    assert "**Conviction: 72/100**" in transcript

    # Bear saw the bull's argument; the judge saw both.
    assert "bull case" in calls[1][1][0]
    assert "bull case" in calls[2][1][0] and "bear case" in calls[2][1][0]


def test_run_debate_is_idempotent(conn, monkeypatch) -> None:
    _seed_thesis(conn)
    _staged_batches(
        monkeypatch,
        [
            [Argument(argument="b")],
            [Argument(argument="r")],
            [Verdict(conviction=50, reasoning="even")],
            [], [], [],  # second run: three stages over zero pending theses
        ],
    )
    assert debate.run_debate(["AAPL"]) == {"AAPL": 1}
    assert debate.run_debate(["AAPL"]) == {"AAPL": 0}


def test_run_debate_drops_thesis_when_a_stage_fails(conn, monkeypatch) -> None:
    """A None at the bull stage (e.g. no key) yields no row and no transcript."""
    _seed_thesis(conn)
    _staged_batches(monkeypatch, [[None], [], []])

    assert debate.run_debate(["AAPL"]) == {"AAPL": 0}
    check = db.connect(config.DB_PATH)
    assert check.execute("SELECT COUNT(*) FROM debates").fetchone()[0] == 0
    check.close()
    assert not (config.DEBATES_DIR / "AAPL_2026-01-29.md").exists()


def test_verdict_rejects_out_of_range_conviction() -> None:
    with pytest.raises(Exception):
        Verdict(conviction=101, reasoning="too sure")


def test_load_convictions_orders_by_date(conn, monkeypatch) -> None:
    _seed_thesis(conn, "AAPL", "2026-01-29")
    _seed_thesis(conn, "NVDA", "2025-11-19")
    _staged_batches(
        monkeypatch,
        [
            [Argument(argument="b")], [Argument(argument="r")],
            [Verdict(conviction=60, reasoning="ok")],
            [Argument(argument="b")], [Argument(argument="r")],
            [Verdict(conviction=40, reasoning="meh")],
        ],
    )
    debate.run_debate(["AAPL", "NVDA"])
    rows = debate.load_convictions(conn)
    assert [(r["ticker"], r["conviction"]) for r in rows] == [("NVDA", 40), ("AAPL", 60)]
