"""Unit tests for EDGAR ingestion — all network access is faked.

The fake payloads mirror the real API shapes verified against live EDGAR:
column-oriented parallel arrays, a nested `filings.recent` block in the
main file, and *flat* top-level arrays in paginated older-submission files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from research_desk import db, ingest
from research_desk.ingest import FilingMeta


class FakeResponse:
    def __init__(self, payload: Any = None, content: bytes = b"", status_code: int = 200):
        self._payload = payload
        self.content = content or json.dumps(payload).encode()
        self.status_code = status_code

    def json(self) -> Any:
        return self._payload


def columns(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    """Build EDGAR's parallel-array (column-oriented) shape from row dicts."""
    keys = ["accessionNumber", "form", "filingDate", "reportDate", "primaryDocument"]
    return {k: [r.get(k, "") for r in rows] for k in keys}


def row(accession: str, form: str, filed: str, period: str, doc: str = "doc.htm") -> dict:
    return {
        "accessionNumber": accession,
        "form": form,
        "filingDate": filed,
        "reportDate": period,
        "primaryDocument": doc,
    }


@pytest.fixture
def fake_edgar(monkeypatch):
    """Patch ingest._get to serve canned payloads keyed by URL substring."""
    responses: dict[str, Any] = {}

    def _fake_get(url: str) -> FakeResponse:
        for fragment, resp in responses.items():
            if fragment in url:
                return resp
        raise AssertionError(f"unexpected URL fetched: {url}")

    monkeypatch.setattr(ingest, "_get", _fake_get)
    return responses


# --- lookup_cik ---------------------------------------------------------


def test_lookup_cik_reads_dict_shaped_ticker_map(fake_edgar) -> None:
    fake_edgar["company_tickers.json"] = FakeResponse(
        {
            "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
            "1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        }
    )
    assert ingest.lookup_cik("aapl") == 320193


def test_lookup_cik_unknown_ticker_raises(fake_edgar) -> None:
    fake_edgar["company_tickers.json"] = FakeResponse({"0": {"cik_str": 1, "ticker": "X"}})
    with pytest.raises(KeyError):
        ingest.lookup_cik("ZZZZ")


# --- list_10k_filings ---------------------------------------------------


def test_exact_form_filter_excludes_variants(fake_edgar) -> None:
    """10-K/A, 10-K405, 10-KT, NT 10-K must all be excluded."""
    rows = [
        row("a-1", "10-K", "2025-10-31", "2025-09-27"),
        row("a-2", "10-K/A", "2025-11-15", "2025-09-27"),
        row("a-3", "10-K405", "2001-12-01", "2001-09-29"),
        row("a-4", "10-KT", "2020-03-01", "2019-12-31"),
        row("a-5", "NT 10-K", "2024-12-01", "2024-09-28"),
        row("a-6", "8-K", "2025-08-01", "2025-08-01"),
    ]
    fake_edgar["CIK0000320193.json"] = FakeResponse(
        {"filings": {"recent": columns(rows), "files": []}}
    )
    filings = ingest.list_10k_filings(320193, n_years=5)
    assert [f.accession for f in filings] == ["a-1"]


def test_five_years_from_recent_block_newest_first(fake_edgar) -> None:
    rows = [
        row(f"a-{y}", "10-K", f"{y}-10-30", f"{y}-09-2{y % 10}") for y in range(2019, 2026)
    ]
    fake_edgar["CIK0000320193.json"] = FakeResponse(
        {"filings": {"recent": columns(rows), "files": []}}
    )
    filings = ingest.list_10k_filings(320193, n_years=5)
    assert len(filings) == 5
    assert filings[0].report_date.startswith("2025")
    assert filings[-1].report_date.startswith("2021")


def test_pagination_fallback_uses_flat_shape(fake_edgar) -> None:
    """JPM case: one 10-K in recent, the rest in flat paginated files."""
    recent = [row("r-1", "10-K", "2026-02-13", "2025-12-31")]
    page_rows = [row(f"p-{y}", "10-K", f"{y + 1}-02-14", f"{y}-12-31") for y in range(2020, 2025)]
    fake_edgar["CIK0000019617.json"] = FakeResponse(
        {
            "filings": {
                "recent": columns(recent),
                "files": [
                    {"name": "CIK0000019617-submissions-001.json",
                     "filingFrom": "1994-01-01", "filingTo": "2019-12-31"},
                    {"name": "CIK0000019617-submissions-004.json",
                     "filingFrom": "2020-01-01", "filingTo": "2025-06-30"},
                ],
            }
        }
    )
    # Paginated files: parallel arrays at the TOP level, no wrapper.
    fake_edgar["submissions-004.json"] = FakeResponse(columns(page_rows))
    fake_edgar["submissions-001.json"] = FakeResponse(columns([]))

    filings = ingest.list_10k_filings(19617, n_years=5)
    assert len(filings) == 5
    # Newest page (by filingTo) must have been consulted first.
    assert filings[0].accession == "r-1"
    assert {f.report_date[:4] for f in filings} == {"2021", "2022", "2023", "2024", "2025"}


def test_dedupe_by_report_date_keeps_newest_filing(fake_edgar) -> None:
    rows = [
        row("dup-new", "10-K", "2025-11-05", "2025-09-27"),
        row("dup-old", "10-K", "2025-10-31", "2025-09-27"),
        row("b-1", "10-K", "2024-11-01", "2024-09-28"),
    ]
    fake_edgar["CIK0000320193.json"] = FakeResponse(
        {"filings": {"recent": columns(rows), "files": []}}
    )
    filings = ingest.list_10k_filings(320193, n_years=5)
    assert [f.accession for f in filings] == ["dup-new", "b-1"]


def test_missing_report_date_rows_skipped(fake_edgar) -> None:
    rows = [row("a-1", "10-K", "2025-10-31", "")]
    fake_edgar["CIK0000320193.json"] = FakeResponse(
        {"filings": {"recent": columns(rows), "files": []}}
    )
    assert ingest.list_10k_filings(320193) == []


# --- document_url -------------------------------------------------------


def test_document_url_unpadded_cik_and_dashless_accession() -> None:
    filing = FilingMeta("0000320193-25-000079", "10-K", "2025-10-31", "2025-09-27",
                        "aapl-20250927.htm")
    url = ingest.document_url(320193, filing)
    assert url == (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        "000032019325000079/aapl-20250927.htm"
    )


# --- fetch_filing -------------------------------------------------------


@pytest.fixture
def conn(tmp_path: Path):
    connection = db.connect(tmp_path / "test.db")
    yield connection
    connection.close()


def test_fetch_filing_caches_and_records(fake_edgar, conn, tmp_path: Path) -> None:
    fake_edgar["Archives"] = FakeResponse(content=b"<html>10-K body</html>")
    filing = FilingMeta("0000320193-25-000079", "10-K", "2025-10-31", "2025-09-27",
                        "aapl-20250927.htm")
    path = ingest.fetch_filing(conn, "AAPL", 320193, filing, raw_dir=tmp_path / "raw")

    assert path.read_bytes() == b"<html>10-K body</html>"
    stored = conn.execute("SELECT * FROM filings").fetchone()
    assert stored["ticker"] == "AAPL"
    assert stored["accession"] == filing.accession
    assert stored["size_bytes"] == len(b"<html>10-K body</html>")


def test_fetch_filing_skips_network_when_cached(
    fake_edgar, conn, tmp_path: Path, monkeypatch
) -> None:
    fake_edgar["Archives"] = FakeResponse(content=b"<html>body</html>")
    filing = FilingMeta("0000320193-25-000079", "10-K", "2025-10-31", "2025-09-27",
                        "aapl-20250927.htm")
    ingest.fetch_filing(conn, "AAPL", 320193, filing, raw_dir=tmp_path / "raw")

    def _explode(url: str):
        raise AssertionError("network hit despite cache")

    monkeypatch.setattr(ingest, "_get", _explode)
    path = ingest.fetch_filing(conn, "AAPL", 320193, filing, raw_dir=tmp_path / "raw")
    assert path.exists()


# --- run_ingest graceful failure -----------------------------------------


def test_run_ingest_survives_bad_ticker(fake_edgar, monkeypatch, tmp_path: Path) -> None:
    """One failing ticker logs and yields 0; the run itself never raises."""
    monkeypatch.setattr(ingest.config, "DB_PATH", tmp_path / "t.db")
    fake_edgar["company_tickers.json"] = FakeResponse(
        {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    )
    fake_edgar["CIK0000320193.json"] = FakeResponse(
        {"filings": {"recent": columns([]), "files": []}}
    )
    summary = ingest.run_ingest(["ZZZZ", "AAPL"])
    assert summary == {"ZZZZ": 0, "AAPL": 0}


# --- rate limiter -------------------------------------------------------


def test_throttle_spaces_requests(monkeypatch) -> None:
    """Consecutive requests must be spaced by at least 1/rate seconds."""
    clock = {"now": 0.0}
    sleeps: list[float] = []
    monkeypatch.setattr(ingest.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(ingest.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(ingest, "_last_request_at", 0.0)

    clock["now"] = 0.05  # 50ms after the previous request
    ingest._throttle()
    min_interval = 1.0 / ingest.config.EDGAR_REQUESTS_PER_SEC
    assert sleeps and sleeps[0] == pytest.approx(min_interval - 0.05)
