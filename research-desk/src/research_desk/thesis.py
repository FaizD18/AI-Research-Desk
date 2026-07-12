"""Structured investment theses from risk-change and sentiment signals.

For each ticker and earnings-call quarter, deterministic code assembles the
evidence — year-over-year risk-factor changes from the most recent 10-K filed
*on or before* the call date (no lookahead), plus quarter-over-quarter shifts
in call tone metrics — and one structured LLM call turns that evidence into a
thesis: direction (long / short / neutral), confidence, a short summary, and
evidence items citing the provided sources.

Judging direction and confidence from mixed signals is the judgment task that
justifies the LLM; everything upstream of the prompt is plain SQL and string
formatting. Calls go through the shared cached client (batched, disk-cached)
and degrade gracefully without an API key: quarters are simply left without a
thesis and picked up on the next run.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from research_desk import config, db, llm

log = logging.getLogger(__name__)


class EvidenceItem(BaseModel):
    """One cited piece of supporting evidence."""

    source: str = Field(description="One of the sources named in the prompt, verbatim")
    point: str = Field(description="What this source contributes to the thesis")


class Thesis(BaseModel):
    """Structured LLM verdict for one ticker as of one earnings call."""

    direction: Literal["long", "short", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0, description="0 = coin flip, 1 = high conviction")
    summary: str = Field(description="2-4 sentence thesis grounded only in the evidence")
    evidence: list[EvidenceItem]


_SYSTEM = (
    "You are a buy-side equity analyst producing a point-in-time investment "
    "thesis. You are given only (a) how the company's 10-K risk factors changed "
    "year over year, with severity scores where available, and (b) deterministic "
    "tone metrics from its earnings calls. Output a direction (long / short / "
    "neutral), a confidence between 0 and 1, a 2-4 sentence summary, and evidence "
    "items whose `source` field repeats one of the named sources verbatim. Base "
    "everything ONLY on the evidence provided — do not use any outside knowledge "
    "of the company or of events after the as-of date. If the evidence is weak or "
    "conflicting, say neutral with low confidence rather than inventing a signal."
)


def _latest_filing_before(
    conn: sqlite3.Connection, ticker: str, call_date: str
) -> sqlite3.Row | None:
    """Most recent 10-K filed on or before ``call_date`` that has risk changes."""
    return conn.execute(
        """SELECT f.accession, f.filing_date, f.report_date
           FROM filings f
           WHERE f.ticker = ? AND f.filing_date <= ?
                 AND EXISTS (SELECT 1 FROM risk_changes rc
                             WHERE rc.curr_accession = f.accession)
           ORDER BY f.filing_date DESC LIMIT 1""",
        (ticker, call_date),
    ).fetchone()


def _risk_evidence(conn: sqlite3.Connection, accession: str) -> tuple[str, str]:
    """(source label, formatted risk-change block) for one 10-K comparison."""
    counts = dict(
        conn.execute(
            """SELECT change_type, COUNT(*) FROM risk_changes
               WHERE curr_accession = ? GROUP BY change_type""",
            (accession,),
        ).fetchall()
    )
    top = conn.execute(
        f"""SELECT rc.change_type, rc.paragraph_text, rs.category, rs.severity, rs.rationale
            FROM risk_changes rc
            LEFT JOIN risk_scores rs ON rs.change_id = rc.id
            WHERE rc.curr_accession = ? AND rc.change_type IN ('NEW', 'ESCALATED')
            ORDER BY (rs.severity IS NULL), rs.severity DESC,
                     CASE rc.change_type WHEN 'NEW' THEN 0 ELSE 1 END,
                     rc.similarity ASC
            LIMIT {config.THESIS_TOP_RISKS}""",
        (accession,),
    ).fetchall()

    lines = [
        "- counts: "
        + ", ".join(f"{counts.get(ct, 0)} {ct}" for ct in ("NEW", "ESCALATED", "REMOVED",
                                                           "UNCHANGED")),
        "- notable NEW/ESCALATED risk factors:",
    ]
    for i, row in enumerate(top, 1):
        score = (
            f"{row['category']}, severity {row['severity']}/5 — {row['rationale']}"
            if row["severity"] is not None
            else "unscored"
        )
        excerpt = " ".join(row["paragraph_text"].split())[: config.THESIS_EXCERPT_CHARS]
        lines.append(f"  {i}. [{row['change_type']} | {score}] \"{excerpt}...\"")
    if not top:
        lines.append("  (none — risk language is unchanged from the prior year)")
    return f"10-K {accession}", "\n".join(lines)


def _tone_line(row: sqlite3.Row) -> str:
    return (
        f"FY{row['fiscal_year']}Q{row['fiscal_quarter']} ({row['call_date']}): "
        f"hedging {row['hedging_per_1k']}, uncertainty {row['uncertainty_per_1k']}, "
        f"guidance {row['guidance_per_1k']} per 1k words ({row['word_count']} words)"
    )


def _sentiment_evidence(
    conn: sqlite3.Connection, curr: sqlite3.Row
) -> tuple[str, str]:
    """(source label, formatted tone block) for a call and its prior quarter."""
    prior = conn.execute(
        """SELECT * FROM transcripts
           WHERE ticker = ? AND call_date < ?
           ORDER BY call_date DESC LIMIT 1""",
        (curr["ticker"], curr["call_date"]),
    ).fetchone()

    label = f"FY{curr['fiscal_year']}Q{curr['fiscal_quarter']} earnings call"
    lines = [f"- this quarter: {_tone_line(curr)}"]
    if prior is not None:
        lines.append(f"- prior quarter: {_tone_line(prior)}")
        lines.append(
            "- QoQ change: "
            f"hedging {round(curr['hedging_per_1k'] - prior['hedging_per_1k'], 3):+}, "
            f"uncertainty {round(curr['uncertainty_per_1k'] - prior['uncertainty_per_1k'], 3):+}, "
            f"guidance {round(curr['guidance_per_1k'] - prior['guidance_per_1k'], 3):+}"
        )
    else:
        lines.append("- prior quarter: none available (first ingested quarter)")
    return label, "\n".join(lines)


def build_prompt(conn: sqlite3.Connection, transcript: sqlite3.Row) -> str | None:
    """Assemble the deterministic evidence prompt for one call quarter.

    Returns ``None`` when no 10-K risk comparison exists on or before the call
    date — there is no risk signal to reason over, so no thesis is produced.
    """
    filing = _latest_filing_before(conn, transcript["ticker"], transcript["call_date"])
    if filing is None:
        return None
    risk_source, risk_block = _risk_evidence(conn, filing["accession"])
    tone_source, tone_block = _sentiment_evidence(conn, transcript)
    return (
        f"Ticker: {transcript['ticker']}\n"
        f"As of: {transcript['call_date']}\n\n"
        f"SOURCE \"{risk_source}\" — risk-factor changes vs the prior year "
        f"(filed {filing['filing_date']}, fiscal year ended {filing['report_date']}):\n"
        f"{risk_block}\n\n"
        f"SOURCE \"{tone_source}\" — deterministic tone metrics:\n"
        f"{tone_block}\n\n"
        "Produce the thesis object."
    )


def run_thesis(tickers: list[str]) -> dict[str, int]:
    """Generate theses for every call quarter that doesn't have one yet.

    Returns {ticker: theses stored}. Quarters without a preceding 10-K
    comparison are skipped; quarters whose LLM call fails (e.g. no API key)
    are left for the next run. The pipeline never crashes on a single failure.
    """
    conn = db.connect()
    summary: dict[str, int] = {}
    for ticker in tickers:
        rows = conn.execute(
            """SELECT t.* FROM transcripts t
               LEFT JOIN theses th ON th.transcript_id = t.id
               WHERE t.ticker = ? AND th.id IS NULL
               ORDER BY t.call_date""",
            (ticker.upper(),),
        ).fetchall()

        pending: list[tuple[sqlite3.Row, str]] = []
        for row in rows:
            prompt = build_prompt(conn, row)
            if prompt is None:
                log.info(
                    "%s FY%sQ%s: no 10-K risk data on/before %s; skipping",
                    ticker, row["fiscal_year"], row["fiscal_quarter"], row["call_date"],
                )
                continue
            pending.append((row, prompt))
        if not pending:
            summary[ticker] = 0
            continue

        theses = llm.cached_batch(
            _SYSTEM,
            [p for _, p in pending],
            Thesis,
            max_tokens=config.THESIS_MAX_TOKENS,
        )
        stored = 0
        now = datetime.now(UTC).isoformat()
        for (row, _), result in zip(pending, theses, strict=True):
            if result is None:
                continue
            conn.execute(
                """INSERT OR REPLACE INTO theses
                   (ticker, transcript_id, as_of, direction, confidence, summary,
                    evidence_json, model, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["ticker"], row["id"], row["call_date"], result.direction,
                    result.confidence, result.summary,
                    json.dumps([e.model_dump() for e in result.evidence]),
                    config.LLM_MODEL, now,
                ),
            )
            stored += 1
        conn.commit()
        summary[ticker] = stored
        log.info("%s: stored %d/%d theses", ticker, stored, len(pending))
    conn.close()
    return summary
