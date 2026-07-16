# PROGRESS — AI Research Desk

> Claude: keep this file current. Mark tasks [x] as they complete. Before
> ending ANY session, fill in "Current state" with exactly where things
> stand and what the next action is.

## Current state

- **In progress:** all three phases built and committed (95 fast tests, lint
  clean). Real-data runs done for every key-free stage: 15 filings → 1,767
  risk changes (440 NEW/ESCALATED); 60 call-quarters ingested from the
  user-approved defeatbeta source with tone metrics stored. Key-less
  degradation verified live: `make thesis` queues 40 quarters and skips
  cleanly, `make backtest` warns (no convictions) and exits.
- **Next step:** user adds ANTHROPIC_API_KEY + API credits (~$1–3 one-time),
  then `make score && make thesis && make debate && make backtest`. Then user
  review of the whole project; capture dashboard screenshots for the README
  after the funded run populates it.
- **Blockers / open questions:** ANTHROPIC_API_KEY not set — risk_scores,
  theses, and debates tables empty until the first funded run.

## Phase 1 — RiskDelta pipeline

- [x] Repo structure + extraction strategy (grounded in live EDGAR research)
- [x] Project scaffolding (uv, config.py, db.py, Makefile, pytest setup)
- [x] ingest.py — EDGAR fetch + cache + SQLite metadata
- [x] ingest.py tests passing (16 tests, network faked)
- [x] extract.py — Item 1A parser (ToC-anchor + heading-scan + LLM fallback)
- [x] extract.py tests passing against 3+ real filings (AAPL, NVDA, JPM)
- [x] llm.py — shared disk-cached Anthropic client (used by extract + score)
- [x] diff.py — paragraph embedding + YoY matching + classification
- [x] diff.py tests passing (7 tests; real run: 440 NEW/ESCALATED changes)
- [x] score.py — LLM categorization + severity, batched (50% via Batches API)
      + disk-cached; degrades gracefully without a key
- [x] score.py tests passing (10 tests, client faked — no key/network)
- [~] Phase 1 end-to-end run on all 3 tickers, committed, user reviewed
      (ingest/extract/diff run on real data; scoring gated on ANTHROPIC_API_KEY)

## Phase 2 — Sentiment + thesis

- [x] Transcript source approved (defeatbeta HF dataset, user-approved 2026-07-12)
- [x] transcripts.py + tests (7 tests; real run: 60 call-quarters, 3 tickers × 20)
- [x] thesis.py + tests (8 tests; point-in-time filing selection verified)
- [~] Phase 2 end-to-end, committed, user reviewed
      (transcripts ran on real data; thesis gated on ANTHROPIC_API_KEY)

## Phase 3 — Debate + backtest + dashboard

- [x] debate.py (Bull / Bear / Judge) + logged transcripts (5 tests; stage-batched)
- [x] backtest.py — vectorbt long/short vs SPY, honest failure writeup (8 tests)
- [x] app.py — Streamlit dashboard (AppTest smoke tests, empty + seeded DB)
- [x] README with architecture diagram + limitations section (all 3 phases)
- [ ] Deployed / demo assets captured (screenshots after first funded run)

## Session log

<!-- One line per session: date — what was done — where it stopped -->
2026-07-12 — README fact-checked (53 claims, 3 fixed) + PROGRESS refreshed, wrap-up committed & pushed — Phase 1 blocked only on ANTHROPIC_API_KEY scoring run + user review
2026-07-12/15 — Phases 2+3 built per user request (pay later): transcripts (defeatbeta, user-approved) + thesis + debate + backtest + app; real transcript data ingested (60 quarters); review findings fixed (Sharpe 252d annualization, vectorbt call_seq=auto — probe-confirmed unfilled buys when fully invested, honest report window/turnover labels, call_date validation) + 18 coverage-gap tests → 95 tests; pushed — awaiting user's funded LLM run + review
