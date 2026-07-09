"""Tests for LLM scoring and the shared cached client.

No network or API key: the Anthropic client is faked. These cover the disk
cache, the strict-schema builder, graceful degradation when the client is
unavailable, and the score.py DB wiring.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field, ValidationError

from research_desk import config, db, llm, score
from research_desk.score import RiskScore


class _Out(BaseModel):
    category: str
    severity: int = Field(ge=1, le=5)


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    """Redirect the LLM disk cache to a throwaway directory."""
    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path / "llmcache")


# --- RiskScore / strict schema ------------------------------------------


def test_riskscore_rejects_out_of_range_severity() -> None:
    with pytest.raises(ValidationError):
        RiskScore(category="cyber", severity=7, rationale="too high")


def test_strict_schema_strips_constraints_and_requires_all() -> None:
    schema = llm._strict_schema(_Out)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"category", "severity"}
    assert "minimum" not in schema["properties"]["severity"]
    assert "maximum" not in schema["properties"]["severity"]


# --- cached_call: disk cache --------------------------------------------


def _fake_client(parsed, counter):
    def parse(**kwargs):
        counter.append(1)
        return SimpleNamespace(parsed_output=parsed)

    return SimpleNamespace(messages=SimpleNamespace(parse=parse))


def test_cached_call_writes_then_reads_cache(monkeypatch) -> None:
    parsed = _Out(category="macro", severity=2)
    calls: list[int] = []
    monkeypatch.setattr(llm, "_client", lambda: _fake_client(parsed, calls))

    first = llm.cached_call("sys", "user", _Out)
    second = llm.cached_call("sys", "user", _Out)  # served from disk

    assert first == second == parsed
    assert len(calls) == 1  # second call hit the cache, no API call


def test_cached_call_raises_after_unparseable_retry(monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(llm, "_client", lambda: _fake_client(None, calls))
    with pytest.raises(llm.LLMError):
        llm.cached_call("sys", "user", _Out)
    assert len(calls) == 2  # one retry before giving up


# --- cached_batch --------------------------------------------------------


def test_cached_batch_all_hits_makes_no_client(monkeypatch) -> None:
    parsed = _Out(category="cyber", severity=4)
    calls: list[int] = []
    monkeypatch.setattr(llm, "_client", lambda: _fake_client(parsed, calls))
    llm.cached_call("sys", "u1", _Out)  # populate cache
    calls.clear()

    def _boom():
        raise AssertionError("client must not be constructed on all-hits")

    monkeypatch.setattr(llm, "_client", _boom)
    results = llm.cached_batch("sys", ["u1"], _Out)
    assert results == [parsed]


def test_cached_batch_single_path_below_threshold(monkeypatch) -> None:
    parsed = _Out(category="operational", severity=3)
    calls: list[int] = []
    monkeypatch.setattr(llm, "_client", lambda: _fake_client(parsed, calls))
    results = llm.cached_batch("sys", ["a", "b"], _Out, batch_threshold=8)
    assert results == [parsed, parsed]
    assert len(calls) == 2


def test_cached_batch_degrades_gracefully_without_client(monkeypatch) -> None:
    def _no_key():
        raise RuntimeError("api_key must be set")

    monkeypatch.setattr(llm, "_client", _no_key)
    results = llm.cached_batch("sys", ["x", "y"], _Out, batch_threshold=8)
    assert results == [None, None]  # no crash, unscored items are None


# --- run_score DB wiring -------------------------------------------------


def _seed_changes(conn) -> None:
    conn.execute(
        """INSERT INTO filings (accession, ticker, cik, form, filing_date,
           report_date, primary_document, doc_url, local_path, sha256,
           size_bytes, fetched_at) VALUES
           ('p','NVDA',1,'10-K','2024-02-01','2024-01-28','d','u','l','s',1,'t'),
           ('c','NVDA',1,'10-K','2025-02-01','2025-01-26','d','u','l','s',1,'t')"""
    )
    rows = [
        ("NEW", "a new cyber risk"),
        ("ESCALATED", "an escalated macro risk"),
        ("UNCHANGED", "an unchanged risk"),  # must NOT be scored
        ("REMOVED", "a removed risk"),  # must NOT be scored
    ]
    for i, (ct, text) in enumerate(rows):
        conn.execute(
            """INSERT INTO risk_changes (ticker, prior_accession, curr_accession,
               change_type, similarity, paragraph_text, matched_text, paragraph_index)
               VALUES ('NVDA','p','c',?,?,?,NULL,?)""",
            (ct, 0.8 if ct != "NEW" else None, text, i),
        )
    conn.commit()


def test_run_score_scores_only_new_and_escalated(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "s.db")
    conn = db.connect(config.DB_PATH)
    _seed_changes(conn)
    conn.close()

    def fake_batch(system, users, model_cls, **kwargs):
        # Exactly the NEW + ESCALATED prompts should reach the scorer.
        assert len(users) == 2
        return [
            RiskScore(category="cyber", severity=4, rationale="new cyber threat"),
            RiskScore(category="macro", severity=3, rationale="macro deterioration"),
        ]

    monkeypatch.setattr(llm, "cached_batch", fake_batch)

    assert score.run_score(["NVDA"]) == {"NVDA": 2}
    conn = db.connect(config.DB_PATH)
    stored = conn.execute(
        """SELECT rc.change_type, rs.category, rs.severity FROM risk_scores rs
           JOIN risk_changes rc ON rc.id = rs.change_id ORDER BY rc.change_type"""
    ).fetchall()
    conn.close()
    assert {r["change_type"] for r in stored} == {"ESCALATED", "NEW"}
    assert all(1 <= r["severity"] <= 5 for r in stored)


def test_run_score_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "s.db")
    conn = db.connect(config.DB_PATH)
    _seed_changes(conn)
    conn.close()

    monkeypatch.setattr(
        llm,
        "cached_batch",
        lambda s, users, m, **k: [
            RiskScore(category="cyber", severity=2, rationale="r") for _ in users
        ],
    )
    assert score.run_score(["NVDA"]) == {"NVDA": 2}
    # Second run: everything already scored, no work left.
    assert score.run_score(["NVDA"]) == {"NVDA": 0}


def test_run_score_skips_unscorable_items(tmp_path, monkeypatch) -> None:
    """A None from the scorer (e.g. missing key) yields no row and no crash."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "s.db")
    conn = db.connect(config.DB_PATH)
    _seed_changes(conn)
    conn.close()

    monkeypatch.setattr(llm, "cached_batch", lambda s, users, m, **k: [None, None])
    assert score.run_score(["NVDA"]) == {"NVDA": 0}
    conn = db.connect(config.DB_PATH)
    assert conn.execute("SELECT COUNT(*) FROM risk_scores").fetchone()[0] == 0
    conn.close()
