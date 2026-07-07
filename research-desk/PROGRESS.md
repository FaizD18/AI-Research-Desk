# PROGRESS — AI Research Desk

> Claude: keep this file current. Mark tasks [x] as they complete. Before
> ending ANY session, fill in "Current state" with exactly where things
> stand and what the next action is.

## Current state

- **In progress:** (nothing yet — project not started)
- **Next step:** Read SPEC.md, propose repo structure + Item 1A extraction
  strategy + transcript data source, wait for approval
- **Blockers / open questions:** none

## Phase 1 — RiskDelta pipeline

- [ ] Repo structure + extraction strategy proposed and approved
- [ ] Project scaffolding (uv, config.py, Makefile, pytest setup)
- [ ] ingest.py — EDGAR fetch + cache + SQLite metadata
- [ ] ingest.py tests passing
- [ ] extract.py — Item 1A parser (heuristics + LLM fallback)
- [ ] extract.py tests passing against 3+ real filings (AAPL, NVDA, JPM)
- [ ] diff.py — paragraph embedding + YoY matching + classification
- [ ] diff.py tests passing
- [ ] score.py — LLM categorization + severity, batched + cached
- [ ] score.py tests passing
- [ ] Phase 1 end-to-end run on all 3 tickers, committed, user reviewed

## Phase 2 — Sentiment + thesis

- [ ] Transcript source approved
- [ ] transcripts.py + tests
- [ ] thesis.py + tests
- [ ] Phase 2 end-to-end, committed, user reviewed

## Phase 3 — Debate + backtest + dashboard

- [ ] debate.py (Bull / Bear / Judge) + logged transcripts
- [ ] backtest.py — vectorbt long/short vs SPY, honest failure writeup
- [ ] app.py — Streamlit dashboard
- [ ] README with architecture diagram + limitations section
- [ ] Deployed / demo assets captured

## Session log

<!-- One line per session: date — what was done — where it stopped -->
