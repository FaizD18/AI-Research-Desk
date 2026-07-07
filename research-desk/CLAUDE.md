# CLAUDE.md — AI Research Desk

## Project

Multi-agent SEC filing analysis system: EDGAR 10-K risk extraction →
year-over-year diffing → LLM scoring → sentiment signals → agent debate →
backtest. Full plan lives in SPEC.md — read it before any work.
Current state lives in PROGRESS.md — read it at the start of every session.

## Stack

- Python 3.11+, dependencies managed with `uv` (never pip directly)
- SQLite for all persistence (no external DB)
- sentence-transformers for embeddings, Anthropic API (claude-sonnet) for scoring
- vectorbt for backtesting, Streamlit for the dashboard, pytest for tests

## Hard rules

- LLM calls ONLY where judgment is required (categorization, severity,
  debate). Never for section-boundary detection, math, date parsing, or
  anything deterministic code can do
- Every LLM call goes through the shared cached client in `llm.py` — no
  raw `anthropic` calls scattered in modules. Cache key = hash of
  (model, system, messages). Reruns must cost ~$0
- SEC EDGAR: declared User-Agent header with contact email, max 10
  requests/sec, cache every raw filing to `data/raw/` before parsing
- One bad filing logs a warning and skips — the pipeline never crashes on
  a single document
- API keys from environment variables only (`ANTHROPIC_API_KEY`)
- Never fabricate or cherry-pick backtest numbers. Report what the data
  shows, including failures

## Conventions

- Type hints and docstrings on every public function
- All thresholds/tickers/model names in `config.py` — nothing inline
- Tests live in `tests/`, mirror module names (`test_extract.py`)
- Commit after each module passes tests; message format:
  `phase1: extract.py — Item 1A parser + tests for AAPL/NVDA/JPM`
- Before ending any session: update PROGRESS.md with current task state
  and the exact next step

## Style

- Prefer small pure functions over classes unless state is genuinely needed
- No premature abstraction — three concrete uses before a helper
