# AI Research Desk

A multi-agent system that reads SEC filings and earnings calls, turns them into
investment theses, has agents debate those theses, and backtests the resulting
signals. Built as a portfolio project — the priorities are code quality, a clean
architecture, and **honest evaluation of what works and what doesn't**.

**All three phases are built and tested.** Every deterministic stage has run on
real data; the three LLM-gated stages (risk scoring, theses, debates) are
implemented, cached, and awaiting their first funded run — see *Running the LLM
stages* at the bottom.

---

## Phase 1 — RiskDelta

RiskDelta answers a specific question: *which risks did a company newly disclose
or materially escalate in its latest 10-K, and how severe are they?* It pulls
five years of 10-Ks per ticker from SEC EDGAR, extracts the Item 1A "Risk
Factors" section, diffs the risk language year over year, and scores the new and
escalated risks with an LLM.

```
 ticker
   │
   ▼
┌──────────────┐   last 5y of 10-Ks (declared UA, ≤10 req/s, cached)
│  ingest.py   │──────────────────────────────────────────────┐
└──────────────┘                                               │
   │  data/raw/<ticker>/<accession>.html  + filings table      │
   ▼                                                           SEC EDGAR
┌──────────────┐   Item 1A via ToC-anchor / heading-scan        (data.sec.gov,
│  extract.py  │   heuristics; LLM only to pick boundaries        www.sec.gov)
└──────────────┘   when the deterministic gate fails ───────────┘
   │  data/extracted/<accession>.txt  + extractions table
   ▼
┌──────────────┐   embed risk paragraphs (bge-small), match
│   diff.py    │   across consecutive years by cosine similarity
└──────────────┘   → NEW / REMOVED / ESCALATED / UNCHANGED
   │  risk_changes table
   ▼
┌──────────────┐   LLM scores NEW/ESCALATED paragraphs:
│  score.py    │   category + 1–5 severity + rationale
└──────────────┘   (batched, disk-cached; llm.py)
   │  risk_scores table
   ▼
 SQLite (data/research_desk.db) — every stage reads the previous stage's table,
 so any stage can be re-run independently and inspected with plain SQL.
```

**One LLM boundary, on purpose.** Everything that can be done with a rule is done
with a rule: section location, paragraph matching, and change classification are
all deterministic. The LLM is used in exactly two places, both requiring
judgment: assigning severity to a risk paragraph (`score.py`), and — only when
the deterministic extractor fails its own validation — picking two *boundary
indexes* from an enumerated heading outline (`extract.py`). Even in that
fallback the model never sees or generates section prose, so it cannot inject
text into the pipeline.

### Quickstart

Requires [uv](https://docs.astral.sh/uv/). The LLM scoring stage needs
`ANTHROPIC_API_KEY`; every other stage runs without any key.

```bash
make setup                 # uv sync — create .venv, install deps
make ingest                # fetch AAPL/NVDA/JPM 10-Ks (last 5y) into data/
make analyze               # extract Item 1A → YoY diff → LLM scoring
make transcripts           # fetch 5y of earnings-call transcripts + tone metrics
make thesis                # LLM thesis per ticker/quarter (needs ANTHROPIC_API_KEY)
make debate                # Bull/Bear/Judge conviction per thesis (needs key)
make backtest              # monthly long/short vs SPY from conviction scores
make app                   # Streamlit dashboard
make test                  # fast tests (network + model download excluded)
make test-all              # everything, incl. the real-filing / embedding tests
```

Individual stages: `make extract`, `make diff`, `make score`. Tickers, cosine
thresholds, model names, and extraction gates are all configured in
[`src/research_desk/config.py`](src/research_desk/config.py).

### Example output

Real results from the three seed tickers (AAPL, NVDA, JPM), taken verbatim from
the pipeline. RiskDelta surfaces genuinely material year-over-year changes:

**NVDA FY2026 — a NEW risk** (no close match in the prior year's 10-K):

> We are finalizing an investment and partnership agreement with OpenAI. There
> is no assurance that we will enter into an investment and partnership
> agreement with OpenAI or that a transaction will be completed.

**JPM FY2023 — a REMOVED risk** (present in the FY2022 10-K, dropped as the
LIBOR transition completed):

> …a particular alternative reference rate will be widely accepted by market
> participants, or that market acceptance of that rate will not be hindered by
> the introduction of other reference rates.

**AAPL — an ESCALATED risk** (retained but materially reworded, cosine 0.73):

> Many governments, regulators, investors, employees, customers and other
> stakeholders are increasingly focused on environmental, social and governance
> considerations…

Change counts across the 12 year-over-year comparisons:

| Ticker | NEW | ESCALATED | REMOVED | UNCHANGED |
|--------|----:|----------:|--------:|----------:|
| AAPL   |   0 |        33 |       0 |       338 |
| NVDA   |   4 |       173 |       0 |       317 |
| JPM    |   6 |       224 |       5 |       667 |

The shape is itself informative: Apple's risk language is remarkably stable
year to year; NVIDIA's was heavily reworded across the AI build-out; JPMorgan's
bank-risk boilerplate is voluminous and highly self-similar. (See *Limitations*
for what that self-similarity does to the classifier.)

---

## Phase 2 — Sentiment + thesis

```
┌───────────────┐   last 5y of earnings calls, from the openly published
│transcripts.py │   defeatbeta dataset on Hugging Face (no scraping, no key);
└───────────────┘   cached JSON → deterministic tone metrics per quarter:
   │                hedging / uncertainty / guidance terms per 1k words
   │  transcripts table
   ▼
┌───────────────┐   risk changes from the latest 10-K filed ON OR BEFORE the
│   thesis.py   │   call date + QoQ tone deltas → one structured LLM call →
└───────────────┘   direction (long/short/neutral), confidence, cited evidence
   │  theses table
```

Tone metrics are word-list counts (Loughran-McDonald-inspired, lists in
`config.py`) over non-operator speech — counting words is not a judgment task,
so no LLM is involved. Levels vary by speaker style; the thesis prompt uses
quarter-over-quarter *shifts*.

Real example from the ingested data (60 call-quarters): JPMorgan's
2025-04-11 call — the first after the April 2025 tariff shock — shows the
largest tone shift in the dataset, **+3.68 uncertainty terms per 1k words**
quarter-over-quarter.

## Phase 3 — Debate + backtest + dashboard

```
┌───────────────┐   Bull argues each thesis, Bear attacks it using the same
│   debate.py   │   evidence, Judge scores conviction 0-100; full transcripts
└───────────────┘   logged to data/debates/ (stage-wise batched LLM calls)
   │  debates table
   ▼
┌───────────────┐   conviction → monthly-rebalanced 1/N long/short, applied
│  backtest.py  │   only AFTER each signal's call date; vectorbt vs SPY with
└───────────────┘   fees → data/backtest_report.md incl. failure writeup
   │
   ▼
┌───────────────┐   Streamlit: risk timeline, call tone, thesis history,
│    app.py     │   debate transcripts, backtest report — works at any
└───────────────┘   pipeline stage, including before any LLM spend
```

A thesis trades only when the Judge found it convincing (conviction ≥ 60): a
convincing long thesis gets a long weight, a convincing short thesis a short
weight, everything else stays flat. Weights are 1/N of the universe so gross
exposure never exceeds 100%.

---

## Design principles

- **Deterministic first.** LLMs are used only for genuine judgment calls
  (severity scoring; boundary selection as a *validated fallback*). Section
  detection, paragraph matching, dates, and math are all code.
- **Everything cached.** Raw filings, extracted sections, paragraph embeddings,
  and every LLM response are cached to disk. Re-running the analysis pipeline
  (`make analyze`) makes zero network calls and costs ~$0; re-running ingest
  re-fetches only lightweight EDGAR metadata and never re-downloads a cached
  filing.
- **Graceful failure.** One malformed filing, failed fetch, or unscorable
  paragraph logs a warning and is skipped — a single bad input never crashes a
  run. Filings that fail extraction validation are quarantined, not emitted.
- **Point-in-time by construction.** Every filing stores its public
  `filing_date` separately from its fiscal `report_date`, so the eventual
  backtest (Phase 3) can apply a signal only *after* it became public — no
  lookahead.
- **No secrets in code.** The API key is read from the environment only; the
  repo ships a `.env.example`, and `data/` (filings, DB, caches) is gitignored.

## Tech stack

Python 3.11+ (managed with `uv`), SQLite, `lxml`, `requests`,
`sentence-transformers` (BAAI/bge-small-en-v1.5), the Anthropic API
(`claude-sonnet-5`, structured outputs + Message Batches), `pydantic`,
`defeatbeta-api` (earnings transcripts), `yfinance` (prices), `vectorbt`
(backtest), `streamlit` (dashboard), and `pytest`. Type hints and docstrings
on public functions; a `Makefile` wraps every stage.

## Testing

`make test` runs the fast suite (95 tests) with all network and model access
faked, so it needs no key, no data, and no downloads. Extraction is tested
against synthetic fixtures that reproduce each filer's *real* HTML structure
(Apple's four-`&nbsp;` bold heading, NVIDIA's green heading with cross-references
that precede the real one, JPMorgan's non-bold heading with a trailing period
and interleaved page artifacts). Phase 2/3 tests fake the LLM client, the
transcript source, and market data: tone metrics run over fixture payloads
shaped exactly like the real cache files, the backtest runs on synthetic
prices rigged so correct point-in-time behavior is distinguishable from
lookahead, and the dashboard is smoke-tested with Streamlit's `AppTest` on
both empty and seeded databases. `make test-all` adds `slow` tests that run
the parser over all 15 real cached filings and embed real risk paragraphs.

## Limitations

Honest about what this does and doesn't do:

- **"ESCALATED" is a language signal, not a confirmed escalation.** Any risk
  paragraph whose best year-over-year cosine match lands in `[0.70, 0.95)` is
  labeled ESCALATED, meaning "retained but materially reworded." A reword can be
  neutral or even a *de*-escalation; the direction and magnitude are what the
  LLM severity score is for. On heavily-reworded filings (NVIDIA) this band is
  large and includes cosmetic edits.
- **REMOVED is under-detected for homogeneous risk corpora.** A prior-year
  paragraph is REMOVED only if *nothing* in the current year matches it above
  0.70. Bank risk-factor language (JPMorgan) is so self-similar that a genuinely
  dropped risk often still has a near-neighbor, so it reads as UNCHANGED. The
  matching is greedy and independent (not a one-to-one assignment), which also
  lets several current paragraphs share one prior match.
- **Thresholds are not tuned or validated.** The 0.70 / 0.95 cosine cutoffs are
  reasonable defaults, not fit to labeled data. They were deliberately *not*
  tuned on these tickers, so no accuracy is claimed.
- **Scoring requires a key and hasn't been run at scale here.** Without
  `ANTHROPIC_API_KEY` the 440 NEW/ESCALATED changes are left unscored (by
  design). The scoring prompt and schema are implemented and cache-tested, but
  the severity numbers are only as good as a single-paragraph LLM judgment with
  no cross-filing context.
- **Coverage is three tickers.** Extraction is verified on AAPL/NVDA/JPM
  (2021–2026), which already span the hard cases (huge bank filing, iXBRL,
  format drift), but a broader S&P-100 sweep (Phase 3's backtest universe) will
  surface filings the two heuristics miss and push work onto the LLM fallback.
- **Single small embedding model.** bge-small-en-v1.5 is fast and CPU-friendly
  but has a 512-token window; unusually long risk paragraphs are truncated
  before embedding.
- **Tone metrics count words, not meaning.** "Risk" scores the same in
  "de-risked the portfolio"; negation, irony, and context are invisible.
  Using QoQ shifts rather than levels mitigates speaker-style bias but not
  the underlying naivety — that's the price of keeping sentiment deterministic.
- **The LLM knows the future.** Theses and debates come from a model whose
  training data overlaps the backtest window. Prompts restrict it to the
  provided point-in-time evidence, but knowledge leakage through the model
  cannot be ruled out — treat any backtest outperformance as suspect. This
  is documented in every generated backtest report.
- **Three tickers is an architecture demo, not a strategy.** Nothing the
  backtest reports is statistically significant, and the tickers were chosen
  today (survivorship). Conviction thresholds and 1/N sizing are deliberate
  defaults, fixed before any results were computed and never tuned.
- **Debate agents argue over a closed record.** Bull, Bear, and Judge see only
  the thesis's cited evidence — by design, so conviction measures how well the
  thesis survives scrutiny of its own evidence, not who can fetch better facts.
- **Transcript provenance.** The defeatbeta dataset is a community-maintained
  mirror; call dates are taken from its `report_date` field, and transcripts
  may be edited or partial. A commercial pipeline would license IR transcripts
  directly.

---

## Running the LLM stages (first funded run)

Everything above the LLM boundary has already run on real data: 15 filings
parsed and diffed, 60 call-quarters ingested and scored for tone. The three
LLM stages need `ANTHROPIC_API_KEY` (see `.env.example`) and run in order:

```bash
make score      # ~440 risk paragraphs
make thesis     # ~40 call-quarters with a preceding 10-K
make debate     # 3 batched calls per thesis (Bull, Bear, Judge)
make backtest   # now has conviction signals to trade
```

All calls go through the Message Batches API (50% discount) and a disk cache,
so the whole first run costs roughly **$1–3 one-time** at current pricing, and
every rerun is $0. Without a key each stage logs what it skipped and leaves
the work queued for the next run — nothing crashes.
