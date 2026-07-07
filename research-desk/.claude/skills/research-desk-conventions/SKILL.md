---
name: research-desk-conventions
description: Conventions and hard rules for the AI Research Desk project (SEC EDGAR ingestion, Item 1A extraction, embedding-based diffing, LLM scoring, agent debate, vectorbt backtesting). Always use this skill when working anywhere in this repository — writing or editing any module, adding tests, touching EDGAR/network code, making LLM calls, or running backtests, even for small fixes.
---

# Research Desk Conventions

## EDGAR access (non-negotiable)

- Every request carries `User-Agent: FaizResearchDesk admin@example.com`
  style header (real contact email from config, per SEC fair-access policy)
- Rate limit: hard cap 10 requests/sec, implemented centrally in
  `ingest.py`, not per-call-site
- Cache raw filing HTML to `data/raw/<ticker>/<accession>.html` before any
  parsing. Parsers read from cache, never from the network

## Item 1A extraction strategy

- Heuristics first: locate section via regex over normalized text
  ("Item 1A", "Risk Factors") with boundary = next "Item 1B" / "Item 2"
  marker; handle table-of-contents false positives by requiring minimum
  section length
- LLM fallback ONLY when heuristics return nothing or fail validation
  (section too short/long). Log every fallback — a rising fallback rate
  means the heuristics need work
- Every extraction change must keep tests passing for AAPL, NVDA, JPM
  fixtures in `tests/fixtures/`

## LLM usage

- All calls through `llm.py::cached_call()` — disk cache keyed on
  (model, system, messages hash). Never a raw client elsewhere
- Structured outputs: request JSON, validate with a dataclass/pydantic
  model, retry once on parse failure, then log and skip
- Model name lives in `config.py` only
- Debate transcripts are saved in full to `data/debates/` — they are a
  demo artifact, not disposable

## Backtesting honesty

- Signals must be point-in-time: a filing's signal applies only AFTER its
  public filing date (no lookahead)
- Report Sharpe, max drawdown, and turnover vs SPY benchmark; include
  periods where the strategy loses
- Never tune thresholds on the full test period and report that as
  out-of-sample

## Workflow

- Read PROGRESS.md at session start; update it before session end
- Commit at every green-test checkpoint
- Deterministic code over LLM calls wherever a rule can be written
