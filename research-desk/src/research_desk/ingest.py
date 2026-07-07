"""Fetch 10-K filings from SEC EDGAR and cache them locally.

Data flow: ticker -> CIK (company_tickers.json) -> filing metadata
(data.sec.gov submissions API) -> primary 10-K documents (www.sec.gov
Archives) -> data/raw/<ticker>/<accession>.html + a row in the `filings`
table. Downstream stages read only from the local cache, never the network.

EDGAR specifics this module encodes (verified against the live API):
- Both hosts share one fair-access budget; a single module-level rate
  limiter throttles every request and every request carries the declared
  User-Agent from config.
- `filings.recent` covers years for most filers, but heavy filers (JPM
  files ~2k structured-product notices a month) need the paginated
  older-submission files, which use a *flat* JSON shape with no
  `filings.recent` wrapper.
- Form must equal "10-K" exactly: prefix matching would wrongly include
  10-K/A amendments (usually no Item 1A), 10-KT, and legacy 10-K405.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from research_desk import config, db

log = logging.getLogger(__name__)


# --- Rate-limited EDGAR access ------------------------------------------

_session = requests.Session()
_session.headers.update(
    {"User-Agent": config.EDGAR_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
)
_last_request_at = 0.0

_RETRYABLE_STATUSES = {403, 429, 500, 502, 503}


def _throttle() -> None:
    """Block until the next request is allowed under the shared rate limit."""
    global _last_request_at
    min_interval = 1.0 / config.EDGAR_REQUESTS_PER_SEC
    wait = _last_request_at + min_interval - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def _get(url: str) -> requests.Response:
    """GET with the declared User-Agent, rate limiting, and retry/backoff.

    Retries 403/429/5xx (EDGAR uses 403 for rate-limit blocks) with
    exponential backoff; raises for anything else or after the last attempt.
    """
    for attempt in range(config.EDGAR_MAX_RETRIES + 1):
        _throttle()
        response = _session.get(url, timeout=60)
        if response.status_code == 200:
            return response
        if response.status_code in _RETRYABLE_STATUSES and attempt < config.EDGAR_MAX_RETRIES:
            delay = config.EDGAR_BACKOFF_SECONDS * (2**attempt)
            log.warning(
                "EDGAR %s for %s — retrying in %.0fs (attempt %d/%d)",
                response.status_code, url, delay, attempt + 1, config.EDGAR_MAX_RETRIES,
            )
            time.sleep(delay)
            continue
        response.raise_for_status()
    raise requests.HTTPError(f"gave up on {url} after {config.EDGAR_MAX_RETRIES} retries")


# --- Metadata -----------------------------------------------------------


@dataclass(frozen=True)
class FilingMeta:
    """One 10-K filing as described by the submissions API."""

    accession: str
    form: str
    filing_date: str  # when it became public — the point-in-time signal date
    report_date: str  # fiscal period end
    primary_document: str


def lookup_cik(ticker: str) -> int:
    """Resolve a ticker to its integer CIK via company_tickers.json.

    The file is a dict keyed by stringified index ("0", "1", ...), not a
    list; `cik_str` is an int despite the name. Multiple share classes
    appear as multiple rows with the same CIK, so first match wins.
    """
    payload = _get(config.EDGAR_TICKER_MAP_URL).json()
    for row in payload.values():
        if row["ticker"].upper() == ticker.upper():
            return int(row["cik_str"])
    raise KeyError(f"ticker {ticker!r} not found in EDGAR company_tickers.json")


def _filings_from_columns(block: dict[str, list[Any]]) -> list[FilingMeta]:
    """Convert EDGAR's column-oriented parallel arrays into FilingMeta rows.

    Accepts both metadata shapes: `filings.recent` from the main submissions
    file and the *flat* paginated older-submission files (same arrays, no
    wrapper). Keeps only exact form == "10-K" rows with a report date.
    """
    out: list[FilingMeta] = []
    forms = block.get("form", [])
    for i, form in enumerate(forms):
        if form != "10-K":
            continue
        if not block["reportDate"][i]:
            continue
        out.append(
            FilingMeta(
                accession=block["accessionNumber"][i],
                form=form,
                filing_date=block["filingDate"][i],
                report_date=block["reportDate"][i],
                primary_document=block["primaryDocument"][i],
            )
        )
    return out


def list_10k_filings(cik: int, n_years: int = config.N_YEARS) -> list[FilingMeta]:
    """Return the last `n_years` 10-Ks for a CIK, newest first.

    Reads `filings.recent` first; if that yields fewer than `n_years`
    (heavy filers like JPM keep barely a year in the recent block), walks
    the paginated older-submission files newest-first until satisfied.
    Deduplicates by report date, which guards against same-period refilings.
    """
    main = _get(
        config.EDGAR_SUBMISSIONS_URL.format(filename=f"CIK{cik:010d}.json")
    ).json()
    found = _filings_from_columns(main["filings"]["recent"])

    pages = sorted(
        main["filings"].get("files", []),
        key=lambda f: f.get("filingTo", ""),
        reverse=True,
    )
    for page in pages:
        if len({f.report_date for f in found}) >= n_years:
            break
        older = _get(config.EDGAR_SUBMISSIONS_URL.format(filename=page["name"])).json()
        found.extend(_filings_from_columns(older))

    found.sort(key=lambda f: (f.report_date, f.filing_date), reverse=True)
    deduped: list[FilingMeta] = []
    seen_periods: set[str] = set()
    for filing in found:
        if filing.report_date in seen_periods:
            continue
        seen_periods.add(filing.report_date)
        deduped.append(filing)
    return deduped[:n_years]


# --- Document download ----------------------------------------------------


def document_url(cik: int, filing: FilingMeta) -> str:
    """Archives URL: unpadded CIK + accession without dashes + primary doc name."""
    return config.EDGAR_ARCHIVES_URL.format(
        cik=cik,
        accession_nodash=filing.accession.replace("-", ""),
        document=filing.primary_document,
    )


def fetch_filing(
    conn: Any, ticker: str, cik: int, filing: FilingMeta, raw_dir: Path | None = None
) -> Path:
    """Download one primary document into the cache and record its metadata.

    Skips the network entirely when the file is already cached and recorded.
    """
    raw_dir = raw_dir or config.RAW_DIR
    local_path = raw_dir / ticker.upper() / f"{filing.accession}.html"
    row = conn.execute(
        "SELECT local_path FROM filings WHERE accession = ?", (filing.accession,)
    ).fetchone()
    if row and local_path.exists():
        log.debug("cache hit for %s %s", ticker, filing.accession)
        return local_path

    url = document_url(cik, filing)
    body = _get(url).content
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(body)

    conn.execute(
        """INSERT OR REPLACE INTO filings
           (accession, ticker, cik, form, filing_date, report_date,
            primary_document, doc_url, local_path, sha256, size_bytes, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            filing.accession,
            ticker.upper(),
            cik,
            filing.form,
            filing.filing_date,
            filing.report_date,
            filing.primary_document,
            url,
            str(local_path),
            hashlib.sha256(body).hexdigest(),
            len(body),
            datetime.now(UTC).isoformat(),
        ),
    )
    conn.commit()
    log.info("fetched %s %s (%s, %.1f MB)",
             ticker, filing.accession, filing.report_date, len(body) / 1e6)
    return local_path


def run_ingest(tickers: list[str]) -> dict[str, int]:
    """Ingest the last N years of 10-Ks for each ticker.

    One bad ticker or filing logs a warning and is skipped — the run never
    crashes on a single document. Returns {ticker: filings_cached}.
    """
    conn = db.connect()
    summary: dict[str, int] = {}
    for ticker in tickers:
        try:
            cik = lookup_cik(ticker)
            filings = list_10k_filings(cik)
        except Exception:
            log.exception("skipping %s: could not list filings", ticker)
            summary[ticker] = 0
            continue
        cached = 0
        for filing in filings:
            try:
                fetch_filing(conn, ticker, cik, filing)
                cached += 1
            except Exception:
                log.exception("skipping %s %s: fetch failed", ticker, filing.accession)
        summary[ticker] = cached
        log.info("%s: %d/%d filings cached", ticker, cached, len(filings))
    conn.close()
    return summary
