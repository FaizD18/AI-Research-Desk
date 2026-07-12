# PROGRESS — AI Research Desk

> Claude: keep this file current. Mark tasks [x] as they complete. Before
> ending ANY session, fill in "Current state" with exactly where things
> stand and what the next action is.

## Current state

- **In progress:** Phase 1 wrap-up. All five modules committed and green
  (47 fast tests, lint clean). Real-data run done for ingest → extract → diff:
  15 filings, 1,767 risk changes (440 NEW/ESCALATED). Full README written and
  fact-checked against code and DB by a 7-agent verification pass (53 claims;
  3 corrected: JPM example was FY2023 not FY2025, "zero network calls" holds
  only for `make analyze`, config-centralization overclaim softened).
- **Next step:** set ANTHROPIC_API_KEY and run `make score` to finish the
  end-to-end run (440 paragraphs, batched + disk-cached), then user review of
  Phase 1 before Phase 2 (transcript source also needs user approval).
- **Blockers / open questions:** ANTHROPIC_API_KEY not set in this
  environment — scoring stage unrun, risk_scores table empty.

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
2026-07-12 — README fact-checked (53 claims, 3 fixed) + PROGRESS refreshed, wrap-up committed & pushed — Phase 1 blocked only on ANTHROPIC_API_KEY scoring run + user review
