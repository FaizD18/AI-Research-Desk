"""Streamlit dashboard: risk timeline, sentiment, theses, debates, backtest.

A read-only viewer over the pipeline's SQLite state and disk artifacts —
launch with ``make app``. Every tab handles a stage that hasn't run yet by
pointing at the make target that produces its data, so the dashboard is
useful at any point in the pipeline's life (including before any LLM stage
has been paid for).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from research_desk import config, db


def _query(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def _risk_tab(conn: sqlite3.Connection, ticker: str) -> None:
    changes = _query(
        conn,
        """SELECT f.report_date, rc.change_type, COUNT(*) AS n
           FROM risk_changes rc JOIN filings f ON f.accession = rc.curr_accession
           WHERE rc.ticker = ? GROUP BY f.report_date, rc.change_type""",
        (ticker,),
    )
    if changes.empty:
        st.info("No risk-change data yet — run `make ingest` and `make analyze`.")
        return
    st.subheader("Year-over-year risk-factor changes")
    pivot = (
        changes[changes.change_type != "UNCHANGED"]
        .pivot(index="report_date", columns="change_type", values="n")
        .fillna(0)
    )
    st.bar_chart(pivot)
    st.caption("UNCHANGED paragraphs are omitted from the chart — they dwarf the rest.")

    scored = _query(
        conn,
        """SELECT f.report_date AS fiscal_year_end, rc.change_type, rs.category,
                  rs.severity, rs.rationale,
                  substr(rc.paragraph_text, 1, 200) || '…' AS excerpt
           FROM risk_scores rs
           JOIN risk_changes rc ON rc.id = rs.change_id
           JOIN filings f ON f.accession = rc.curr_accession
           WHERE rc.ticker = ? ORDER BY rs.severity DESC, f.report_date DESC LIMIT 15""",
        (ticker,),
    )
    st.subheader("Highest-severity scored risks")
    if scored.empty:
        st.info("No LLM scores yet — set ANTHROPIC_API_KEY and run `make score`.")
    else:
        st.dataframe(scored, use_container_width=True)


def _sentiment_tab(conn: sqlite3.Connection, ticker: str) -> None:
    tone = _query(
        conn,
        """SELECT call_date, hedging_per_1k, uncertainty_per_1k, guidance_per_1k
           FROM transcripts WHERE ticker = ? ORDER BY call_date""",
        (ticker,),
    )
    if tone.empty:
        st.info("No transcripts yet — run `make transcripts`.")
        return
    st.subheader("Earnings-call tone per 1,000 words")
    st.line_chart(tone.set_index("call_date"))
    st.caption(
        "Deterministic word-list rates (Loughran-McDonald-inspired) over "
        "non-operator speech. Shifts, not levels, are the signal."
    )


def _theses_tab(conn: sqlite3.Connection, ticker: str) -> None:
    theses = _query(
        conn,
        """SELECT as_of, direction, confidence, summary, evidence_json
           FROM theses WHERE ticker = ? ORDER BY as_of DESC""",
        (ticker,),
    )
    if theses.empty:
        st.info("No theses yet — set ANTHROPIC_API_KEY and run `make thesis`.")
        return
    st.subheader("Thesis history")
    st.dataframe(theses.drop(columns=["evidence_json"]), use_container_width=True)
    for _, row in theses.iterrows():
        with st.expander(f"{row.as_of}: {row.direction} ({row.confidence:.2f}) — evidence"):
            for item in json.loads(row.evidence_json):
                st.markdown(f"- **{item['source']}** — {item['point']}")


def _debates_tab(conn: sqlite3.Connection, ticker: str) -> None:
    debates = _query(
        conn,
        """SELECT th.as_of, d.conviction, d.judge_reasoning, d.transcript_path
           FROM debates d JOIN theses th ON th.id = d.thesis_id
           WHERE th.ticker = ? ORDER BY th.as_of DESC""",
        (ticker,),
    )
    if debates.empty:
        st.info("No debates yet — set ANTHROPIC_API_KEY and run `make debate`.")
        return
    choice = st.selectbox("Debate (by call date)", debates.as_of.tolist())
    row = debates[debates.as_of == choice].iloc[0]
    st.metric("Judge conviction", f"{row.conviction}/100")
    st.markdown(f"**Judge reasoning:** {row.judge_reasoning}")
    transcript = Path(row.transcript_path)
    if transcript.exists():
        st.divider()
        st.markdown(transcript.read_text())
    else:
        st.warning(f"Transcript file missing: {transcript}")


def _backtest_tab() -> None:
    report = config.BACKTEST_REPORT_PATH
    if not report.exists():
        st.info("No backtest report yet — run `make backtest` (after `make debate`).")
        return
    st.markdown(report.read_text())


def main() -> None:
    st.set_page_config(page_title="AI Research Desk", layout="wide")
    st.title("AI Research Desk")
    st.caption(
        "10-K risk deltas → earnings-call tone → LLM theses → Bull/Bear/Judge "
        "debates → long/short backtest. All state lives in SQLite; every LLM "
        "stage is disk-cached."
    )

    conn = db.connect()
    try:
        known = _query(conn, "SELECT DISTINCT ticker FROM filings")["ticker"].tolist()
        universe = sorted(set(config.TICKERS) | set(known))
        ticker = st.sidebar.selectbox("Ticker", universe)
        tabs = st.tabs(["Risk timeline", "Sentiment", "Theses", "Debates", "Backtest"])
        with tabs[0]:
            _risk_tab(conn, ticker)
        with tabs[1]:
            _sentiment_tab(conn, ticker)
        with tabs[2]:
            _theses_tab(conn, ticker)
        with tabs[3]:
            _debates_tab(conn, ticker)
        with tabs[4]:
            _backtest_tab()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
