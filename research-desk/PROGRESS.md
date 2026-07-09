# PROGRESS — AI Research Desk

> Claude: keep this file current. Mark tasks [x] as they complete. Before
> ending ANY session, fill in "Current state" with exactly where things
> stand and what the next action is.

## Current state

- **In progress:** extract.py + llm.py complete and committed. Next module is
  diff.py.
- **Next step:** diff.py — split each Item 1A into risk paragraphs, embed with
  BAAI/bge-small-en-v1.5, match paragraphs across consecutive years by cosine
  similarity, classify NEW/REMOVED/ESCALATED/UNCHANGED into risk_changes.
- **Blockers / open questions:** none. All 15 filings (AAPL/NVDA/JPM × 5y)
  extract via the ToC-anchor heuristic (0 LLM fallbacks); word counts 9.5k–19k
  match the research. data/ is gitignored, so real-filing tests are marked
  `slow` and need `make ingest` first; fast tests use committed synthetic
  fixtures reproducing each filer's real HTML structure.

## Phase 1 — RiskDelta pipeline

- [x] Repo structure + extraction strategy (grounded in live EDGAR research)
- [x] Project scaffolding (uv, config.py, db.py, Makefile, pytest setup)
- [x] ingest.py — EDGAR fetch + cache + SQLite metadata
- [x] ingest.py tests passing (16 tests, network faked)
- [x] extract.py — Item 1A parser (ToC-anchor + heading-scan + LLM fallback)
- [x] extract.py tests passing against 3+ real filings (AAPL, NVDA, JPM)
- [x] llm.py — shared disk-cached Anthropic client (used by extract + score)
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
