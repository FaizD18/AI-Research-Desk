"""Earnings-call transcript ingestion and deterministic tone metrics.

Source: the openly published defeatbeta dataset on Hugging Face, queried via
the ``defeatbeta-api`` package (DuckDB over HTTP) — no scraping, no API key.
Every transcript is cached to ``data/transcripts/<ticker>/fy<year>q<q>.json``
before analysis; metric extraction reads only from that cache, mirroring how
EDGAR filings are cached before parsing.

Tone metrics are deterministic word-list counts (word lists in ``config``,
inspired by the Loughran-McDonald financial sentiment dictionaries): hedging
language, uncertainty terms, and guidance mentions, each per 1,000 words so
quarters of different length are comparable. Quarter-over-quarter shifts in
these rates are the sentiment signal consumed by ``thesis.py``. No LLM is
involved — counting words is not a judgment task.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from research_desk import config, db

log = logging.getLogger(__name__)

# The conference operator's paragraphs are call logistics, not management or
# analyst speech; they are excluded from every tone metric.
_OPERATOR_SPEAKER = "operator"

_WORD_RE = re.compile(r"[a-z']+")

# call_date flows straight into point-in-time joins downstream (thesis,
# backtest); a malformed date would corrupt string-comparison ordering.
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _count_words(text: str) -> int:
    """Number of word tokens in ``text`` (lowercased alphabetic runs)."""
    return len(_WORD_RE.findall(text.lower()))


def _term_pattern(terms: list[str]) -> re.Pattern[str]:
    """Compile one alternation matching any term as whole words (phrases ok)."""
    escaped = sorted((re.escape(t.lower()) for t in terms), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b")


_HEDGING = _term_pattern(config.HEDGING_TERMS)
_UNCERTAINTY = _term_pattern(config.UNCERTAINTY_TERMS)
_GUIDANCE = _term_pattern(config.GUIDANCE_TERMS)


def tone_metrics(text: str) -> dict[str, float | int]:
    """Deterministic tone metrics for one transcript's text.

    Returns word_count plus hedging/uncertainty/guidance rates per 1,000
    words. A text below one word returns zero rates rather than dividing
    by zero.
    """
    words = _count_words(text)
    lowered = text.lower()

    def per_1k(pattern: re.Pattern[str]) -> float:
        if words == 0:
            return 0.0
        return round(1000.0 * len(pattern.findall(lowered)) / words, 3)

    return {
        "word_count": words,
        "hedging_per_1k": per_1k(_HEDGING),
        "uncertainty_per_1k": per_1k(_UNCERTAINTY),
        "guidance_per_1k": per_1k(_GUIDANCE),
    }


def _cache_path(ticker: str, fiscal_year: int, fiscal_quarter: int) -> Path:
    return config.TRANSCRIPTS_DIR / ticker.upper() / f"fy{fiscal_year}q{fiscal_quarter}.json"


def fetch_transcripts(ticker: str, *, n_years: int = config.N_YEARS) -> list[Path]:
    """Download and cache the last ``n_years`` of earnings-call transcripts.

    Quarters already on disk are not re-fetched. One bad quarter logs a
    warning and is skipped — a single missing call never fails the ticker.
    Returns the cached file paths (existing and newly written), oldest first.
    """
    # Heavy import with network side effects: only a real fetch needs it.
    from defeatbeta_api.data.ticker import Ticker

    cutoff = (date.today() - timedelta(days=round(365.25 * n_years))).isoformat()
    transcripts = Ticker(ticker.upper()).earning_call_transcripts()
    listing = transcripts.get_transcripts_list()
    listing = listing[listing["report_date"].astype(str) >= cutoff]

    paths: list[Path] = []
    for row in listing.itertuples():
        fy, fq = int(row.fiscal_year), int(row.fiscal_quarter)
        path = _cache_path(ticker, fy, fq)
        if path.exists():
            paths.append(path)
            continue
        try:
            paragraphs = transcripts.get_transcript(fy, fq)
            payload = {
                "ticker": ticker.upper(),
                "fiscal_year": fy,
                "fiscal_quarter": fq,
                "call_date": str(row.report_date)[:10],
                "source": config.TRANSCRIPT_SOURCE,
                "paragraphs": [
                    {"speaker": p.speaker, "content": p.content}
                    for p in paragraphs.itertuples()
                ],
            }
        except Exception:
            log.exception("%s FY%dQ%d: transcript fetch failed; skipping", ticker, fy, fq)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=True))
        paths.append(path)
        log.info("%s FY%dQ%d: cached transcript (%s)", ticker, fy, fq, payload["call_date"])
    return sorted(paths)


def _call_text(payload: dict) -> str:
    """Join every non-operator paragraph into one analyzable text."""
    return "\n".join(
        p["content"]
        for p in payload["paragraphs"]
        if p["speaker"].strip().lower() != _OPERATOR_SPEAKER and p["content"]
    )


def run_transcripts(tickers: list[str]) -> dict[str, int]:
    """Ingest transcripts for ``tickers``: fetch, cache, and store tone metrics.

    Returns {ticker: rows stored}. If the network fetch fails outright the
    stage falls back to whatever is already cached on disk, so reruns and
    offline runs still (re)compute metrics.
    """
    conn = db.connect()
    summary: dict[str, int] = {}
    for ticker in tickers:
        try:
            paths = fetch_transcripts(ticker)
        except Exception:
            log.exception("%s: transcript fetch unavailable; using existing cache", ticker)
            paths = sorted((config.TRANSCRIPTS_DIR / ticker.upper()).glob("fy*.json"))

        stored = 0
        now = datetime.now(UTC).isoformat()
        for path in paths:
            try:
                payload = json.loads(path.read_text())
                metrics = tone_metrics(_call_text(payload))
            except Exception:
                log.exception("%s: unreadable transcript cache %s; skipping", ticker, path.name)
                continue
            if not _DATE_RE.match(str(payload.get("call_date", ""))):
                log.warning(
                    "%s: bad call_date %r in %s; skipping (dates drive point-in-time joins)",
                    ticker, payload.get("call_date"), path.name,
                )
                continue
            if metrics["word_count"] < config.TRANSCRIPT_MIN_WORDS:
                log.warning(
                    "%s FY%sQ%s: only %d words (< %d); skipping stub transcript",
                    ticker, payload["fiscal_year"], payload["fiscal_quarter"],
                    metrics["word_count"], config.TRANSCRIPT_MIN_WORDS,
                )
                continue
            conn.execute(
                """INSERT INTO transcripts
                   (ticker, fiscal_year, fiscal_quarter, call_date, word_count,
                    hedging_per_1k, uncertainty_per_1k, guidance_per_1k,
                    text_path, source, ingested_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (ticker, fiscal_year, fiscal_quarter) DO UPDATE SET
                       call_date = excluded.call_date,
                       word_count = excluded.word_count,
                       hedging_per_1k = excluded.hedging_per_1k,
                       uncertainty_per_1k = excluded.uncertainty_per_1k,
                       guidance_per_1k = excluded.guidance_per_1k,
                       text_path = excluded.text_path,
                       source = excluded.source,
                       ingested_at = excluded.ingested_at""",
                (
                    payload["ticker"], payload["fiscal_year"], payload["fiscal_quarter"],
                    payload["call_date"], metrics["word_count"], metrics["hedging_per_1k"],
                    metrics["uncertainty_per_1k"], metrics["guidance_per_1k"],
                    str(path), payload.get("source", config.TRANSCRIPT_SOURCE), now,
                ),
            )
            stored += 1
        conn.commit()
        summary[ticker] = stored
        log.info("%s: %d transcript quarters stored", ticker, stored)
    conn.close()
    return summary
