# AI Research Desk

A multi-agent system that reads SEC filings and earnings calls, turns them into
investment theses, has agents debate those theses, and backtests the resulting
signals. Built as a portfolio project — the priorities are code quality, a clean
architecture, and **honest evaluation of what works and what doesn't**.

The project ships in three phases. **Phase 1 (RiskDelta) is complete;** Phases 2
and 3 are scoped below.

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
`vectorbt` and `streamlit` (Phase 3), and `pytest`. Type hints and docstrings on
public functions; a `Makefile` wraps every stage.

## Testing

`make test` runs the fast suite (47 tests) with all network and model access
faked, so it needs no key, no data, and no downloads. Extraction is tested
against synthetic fixtures that reproduce each filer's *real* HTML structure
(Apple's four-`&nbsp;` bold heading, NVIDIA's green heading with cross-references
that precede the real one, JPMorgan's non-bold heading with a trailing period
and interleaved page artifacts). `make test-all` adds `slow` tests that run the
parser over all 15 real cached filings and embed real risk paragraphs.

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

---

## Roadmap

- **Phase 2 — Sentiment + thesis.** Ingest earnings-call transcripts (proposed
  source: the openly published `defeatbeta` / HuggingFace dataset — no scraping,
  clean licensing), extract quarter-over-quarter management tone shifts, and
  combine risk-change and sentiment signals into a structured, citation-backed
  thesis per ticker per quarter.
- **Phase 3 — Debate + backtest + dashboard.** A Bull/Bear/Judge agent debate
  producing a 0–100 conviction score with logged transcripts; a monthly-rebalanced
  long/short backtest over the S&P 100 vs SPY (`vectorbt`) with an honest writeup
  of where the signal fails; and a Streamlit dashboard tying risk timelines,
  theses, debates, and backtest results together.
