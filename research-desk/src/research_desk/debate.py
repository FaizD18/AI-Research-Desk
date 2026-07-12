"""Bull / Bear / Judge debate over each thesis.

For every stored thesis, three structured LLM roles run in sequence: a Bull
agent argues the strongest good-faith case that the thesis is right, a Bear
agent attacks it using the same evidence, and a Judge weighs both and returns
a 0-100 conviction that the thesis's stance is correct. Conviction is the
tradable signal consumed by ``backtest.py`` (0 = the thesis is wrong, 50 =
coin flip, 100 = highly convincing).

The stages are batched across theses (all Bulls, then all Bears, then all
Judges) so the Message Batches discount and the disk cache both apply. Full
debate transcripts are written to ``data/debates/`` — they are a demo
artifact, not disposable. A thesis whose debate cannot complete (e.g. no API
key) is simply left undebated and picked up on the next run.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from research_desk import config, db, llm

log = logging.getLogger(__name__)


class Argument(BaseModel):
    """One agent's argument in the debate."""

    argument: str = Field(description="A focused argument grounded only in the evidence")


class Verdict(BaseModel):
    """The Judge's structured ruling on the debate."""

    conviction: int = Field(
        ge=0, le=100, description="0 = thesis is wrong, 50 = coin flip, 100 = compelling"
    )
    reasoning: str = Field(description="2-4 sentences weighing the Bull and Bear cases")


_BULL_SYSTEM = (
    "You are the Bull agent in an investment debate. Argue the strongest "
    "good-faith case that the given thesis (its direction AND its confidence) is "
    "correct. Use only the evidence provided in the thesis object — cite it "
    "specifically. Do not invent facts or use outside knowledge of the company."
)

_BEAR_SYSTEM = (
    "You are the Bear agent in an investment debate. Using ONLY the same "
    "evidence the thesis cites, argue that the thesis is wrong or overconfident: "
    "attack weak inferences, point out what the evidence cannot support, and "
    "propose the strongest alternative reading. Do not invent facts or use "
    "outside knowledge of the company."
)

_JUDGE_SYSTEM = (
    "You are the Judge of an investment debate. You are given a thesis, a Bull "
    "argument for it, and a Bear argument against it. Weigh only what is in "
    "front of you and return a conviction from 0 (the Bear demolished it) to "
    "100 (the Bull case is compelling and survived the Bear's attack), with "
    "2-4 sentences of reasoning. A vague thesis with thin evidence deserves a "
    "conviction near 50 regardless of how confident it sounds."
)


def _thesis_block(row: sqlite3.Row) -> str:
    return (
        f"Ticker: {row['ticker']}\n"
        f"As of: {row['as_of']}\n"
        f"Thesis direction: {row['direction']} (confidence {row['confidence']})\n"
        f"Thesis summary: {row['summary']}\n"
        f"Evidence: {row['evidence_json']}"
    )


def _transcript_path(row: sqlite3.Row) -> Path:
    return config.DEBATES_DIR / f"{row['ticker']}_{row['as_of']}.md"


def _write_transcript(row: sqlite3.Row, bull: str, bear: str, verdict: Verdict) -> Path:
    path = _transcript_path(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# Debate — {row['ticker']} as of {row['as_of']}\n\n"
        f"**Thesis:** {row['direction']} (confidence {row['confidence']})\n\n"
        f"{row['summary']}\n\n"
        f"## Bull\n\n{bull}\n\n"
        f"## Bear\n\n{bear}\n\n"
        f"## Judge\n\n**Conviction: {verdict.conviction}/100**\n\n{verdict.reasoning}\n"
    )
    return path


def run_debate(tickers: list[str]) -> dict[str, int]:
    """Debate every thesis that has no debate yet; returns {ticker: debated}.

    Each stage is batched across the ticker's pending theses. A thesis drops
    out of later stages if an earlier stage failed for it — partial debates
    are never stored, and stage outputs already produced stay in the LLM disk
    cache, so a rerun completes them without re-paying.
    """
    conn = db.connect()
    summary: dict[str, int] = {}
    for ticker in tickers:
        theses = conn.execute(
            """SELECT th.* FROM theses th
               LEFT JOIN debates d ON d.thesis_id = th.id
               WHERE th.ticker = ? AND d.thesis_id IS NULL
               ORDER BY th.as_of""",
            (ticker.upper(),),
        ).fetchall()
        if not theses:
            summary[ticker] = 0
            continue

        bulls = llm.cached_batch(
            _BULL_SYSTEM,
            [_thesis_block(t) for t in theses],
            Argument,
            max_tokens=config.DEBATE_MAX_TOKENS,
        )
        with_bull = [(t, b.argument) for t, b in zip(theses, bulls, strict=True)
                     if b is not None]

        bears = llm.cached_batch(
            _BEAR_SYSTEM,
            [f"{_thesis_block(t)}\n\nBULL ARGUMENT:\n{bull}" for t, bull in with_bull],
            Argument,
            max_tokens=config.DEBATE_MAX_TOKENS,
        )
        with_bear = [(t, bull, b.argument)
                     for (t, bull), b in zip(with_bull, bears, strict=True)
                     if b is not None]

        verdicts = llm.cached_batch(
            _JUDGE_SYSTEM,
            [
                f"{_thesis_block(t)}\n\nBULL ARGUMENT:\n{bull}\n\nBEAR ARGUMENT:\n{bear}"
                for t, bull, bear in with_bear
            ],
            Verdict,
            max_tokens=config.DEBATE_MAX_TOKENS,
        )

        debated = 0
        now = datetime.now(UTC).isoformat()
        for (t, bull, bear), verdict in zip(with_bear, verdicts, strict=True):
            if verdict is None:
                continue
            path = _write_transcript(t, bull, bear, verdict)
            conn.execute(
                """INSERT OR REPLACE INTO debates
                   (thesis_id, conviction, judge_reasoning, transcript_path, model,
                    debated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (t["id"], verdict.conviction, verdict.reasoning, str(path),
                 config.LLM_MODEL, now),
            )
            debated += 1
        conn.commit()
        summary[ticker] = debated
        log.info("%s: debated %d/%d theses", ticker, debated, len(theses))
    conn.close()
    return summary


def load_convictions(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """All debated theses as (ticker, as_of, direction, conviction), oldest first.

    This is the signal surface consumed by the backtest: one signed-able
    conviction per ticker per earnings quarter, dated by the call date it was
    derived from.
    """
    return conn.execute(
        """SELECT th.ticker, th.as_of, th.direction, d.conviction
           FROM debates d JOIN theses th ON th.id = d.thesis_id
           ORDER BY th.as_of"""
    ).fetchall()
