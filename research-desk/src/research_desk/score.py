"""LLM scoring of new and escalated risks.

For every ``NEW`` or ``ESCALATED`` risk-change paragraph, one structured LLM
call assigns a category (regulatory / competitive / macro / operational / cyber
/ litigation / supply-chain), a 1–5 severity, and a one-sentence rationale.
``UNCHANGED`` and ``REMOVED`` paragraphs are not scored — they carry no new risk
signal. All calls go through the shared cached client and are batched, so a
rerun over already-scored changes makes no API calls.

Scoring is the one place in Phase 1 where an LLM is used, because assigning
severity to prose is a judgment task. Everything upstream — section location,
paragraph matching, change classification — is deterministic.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Literal, get_args

from pydantic import BaseModel, Field

from research_desk import config, db, llm

log = logging.getLogger(__name__)

Category = Literal[
    "regulatory",
    "competitive",
    "macro",
    "operational",
    "cyber",
    "litigation",
    "supply-chain",
]

# Keep config and the type in sync — a mismatch would silently drop a category.
assert set(get_args(Category)) == set(config.RISK_CATEGORIES)

SCORED_CHANGE_TYPES = ("NEW", "ESCALATED")


class RiskScore(BaseModel):
    """Structured LLM verdict on one risk-factor paragraph."""

    category: Category
    severity: int = Field(ge=1, le=5, description="1 = minor, 5 = severe/existential")
    rationale: str = Field(description="One sentence justifying the category and severity")


_SYSTEM = (
    "You are a sell-side equity risk analyst. You are given a single risk-factor "
    "paragraph from a company's 10-K that is either newly added this year or was "
    "materially reworded from the prior year. Classify it into exactly one "
    "category from: regulatory, competitive, macro, operational, cyber, "
    "litigation, supply-chain. Assign a severity from 1 (minor / boilerplate) to "
    "5 (severe or potentially existential), judged by how much this risk could "
    "plausibly move the company's fundamentals. Give a one-sentence rationale. "
    "Base the score only on the text provided; do not speculate beyond it."
)


def _prompt(change_type: str, paragraph: str) -> str:
    label = "newly added this year" if change_type == "NEW" else "reworded from last year"
    return f"This risk factor was {label}:\n\n{paragraph}"


def run_score(tickers: list[str]) -> dict[str, int]:
    """Score all unscored NEW/ESCALATED changes for the given tickers.

    Returns {ticker: scored_count}. Items that cannot be scored (e.g. no API
    key) are left unscored and simply picked up on the next run; the pipeline
    never crashes on a scoring failure.
    """
    conn = db.connect()
    placeholders = ",".join("?" for _ in SCORED_CHANGE_TYPES)
    summary: dict[str, int] = {}
    for ticker in tickers:
        rows = conn.execute(
            f"""SELECT rc.id, rc.change_type, rc.paragraph_text
                FROM risk_changes rc
                LEFT JOIN risk_scores rs ON rs.change_id = rc.id
                WHERE rc.ticker = ? AND rc.change_type IN ({placeholders})
                      AND rs.change_id IS NULL
                ORDER BY rc.id""",
            (ticker.upper(), *SCORED_CHANGE_TYPES),
        ).fetchall()
        if not rows:
            summary[ticker] = 0
            continue

        prompts = [_prompt(r["change_type"], r["paragraph_text"]) for r in rows]
        scores = llm.cached_batch(_SYSTEM, prompts, RiskScore)

        scored = 0
        now = datetime.now(UTC).isoformat()
        for row, score in zip(rows, scores, strict=True):
            if score is None:
                continue
            conn.execute(
                """INSERT OR REPLACE INTO risk_scores
                   (change_id, category, severity, rationale, model, scored_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (row["id"], score.category, score.severity, score.rationale,
                 config.LLM_MODEL, now),
            )
            scored += 1
        conn.commit()
        summary[ticker] = scored
        log.info("%s: scored %d/%d unscored changes", ticker, scored, len(rows))
    conn.close()
    return summary
