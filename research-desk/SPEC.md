# SPEC — AI Research Desk

I'm building "AI Research Desk" — a multi-agent system that analyzes SEC
filings and earnings calls, generates investment theses, has agents debate
them, and backtests the resulting signals. I'm a data science student
targeting ML internships. This is a portfolio project: code quality,
architecture, and honest evaluation matter more than raw performance.

BUILD THIS IN 3 PHASES. Complete and test each phase before starting the
next. The user reviews between phases.

---

## PHASE 1: RISK INTELLIGENCE PIPELINE (RiskDelta)

A pipeline that pulls 10-K filings from SEC EDGAR, extracts Item 1A (Risk
Factors), detects year-over-year changes in risk language, and scores
new/escalated risks with an LLM.

Modules:

1. `ingest.py` — given a ticker, fetch last 5 years of 10-Ks from EDGAR
   (proper User-Agent per SEC fair-access rules, rate limit 10 req/sec),
   cache raw HTML locally, store metadata in SQLite
2. `extract.py` — parse Item 1A robustly. Filings vary wildly in HTML
   structure; use parsing heuristics with an LLM fallback ONLY when
   heuristics fail. Write tests against 3+ real filings
3. `diff.py` — split Item 1A into risk paragraphs, embed with
   sentence-transformers, match paragraphs across consecutive years via
   cosine similarity. Classify each: NEW, REMOVED, ESCALATED, UNCHANGED
4. `score.py` — for NEW/ESCALATED risks, LLM call (claude-sonnet,
   structured JSON output) to categorize (regulatory / competitive / macro /
   operational / cyber / litigation / supply-chain) and assign 1-5 severity
   with rationale. Batch and cache all LLM calls

## PHASE 2: SENTIMENT AGENT + THESIS GENERATION

5. `transcripts.py` — ingest earnings call transcripts for the same tickers
   (use a free source; propose options and tradeoffs before implementing).
   Extract management tone shifts quarter-over-quarter: hedging language,
   uncertainty terms, guidance changes
6. `thesis.py` — combine risk-change signals + sentiment signals into a
   structured thesis object per ticker per quarter: direction (long / short /
   neutral), confidence, supporting evidence with citations back to source
   filings

## PHASE 3: DEBATE + BACKTEST

7. `debate.py` — three-agent structure: a Bull agent argues for the thesis,
   a Bear agent argues against using the same evidence, a Judge agent
   scores the debate and outputs a final conviction score (0-100) with
   reasoning. Log full debate transcripts for inspection
8. `backtest.py` — turn conviction scores into a simple monthly-rebalanced
   long/short strategy, backtest over 5 years of S&P 100 tickers using
   vectorbt, benchmark against SPY. Report Sharpe, max drawdown, and —
   critically — an honest writeup of where the signal fails
9. `app.py` — Streamlit dashboard: per-ticker risk timeline, thesis
   history, debate transcripts, and backtest results

---

## TECH STACK

Python 3.11+ with uv, SQLite, sentence-transformers, Anthropic API,
vectorbt, Streamlit, pytest. Config in one file. Type hints and docstrings
everywhere. Makefile with: setup, ingest, analyze, debate, backtest, app.

## DESIGN RULES

- Deterministic code wherever possible; LLM calls only where judgment is
  required. Never use an LLM for section-boundary detection, math, or
  anything parseable
- Every LLM call cached to disk so full-pipeline reruns cost ~$0
- Graceful failure: one bad filing logs and skips, never crashes the run
- README with architecture diagram, setup, example output, and a
  "limitations" section

## WORKFLOW RULES

- Maintain PROGRESS.md: mark tasks done as you complete them, and record
  the current in-progress task and next step before ending any work
- Commit at every working checkpoint (module passes tests), with clear
  commit messages — not just at phase ends
- START BY: proposing the repo structure, your Item 1A extraction
  strategy, and your recommended earnings-transcript data source. Wait for
  approval before writing code

## DO NOT

- Hardcode API keys
- Scrape without rate limits
- Fabricate backtest results or claim performance the data doesn't support

Initial test tickers: AAPL, NVDA, JPM.
