# AI Research Desk

Multi-agent system that analyzes SEC 10-K filings and earnings calls,
generates investment theses, debates them with Bull/Bear/Judge agents, and
backtests the resulting signals.

**Status: Phase 1 (RiskDelta pipeline) under construction.** Full README with
architecture diagram, example output, and limitations lands at the end of
Phase 1. See `SPEC.md` for the plan and `PROGRESS.md` for current state.

## Quickstart

```bash
make setup    # uv sync — creates .venv and installs everything
make ingest   # fetch last 5y of 10-Ks for AAPL, NVDA, JPM from SEC EDGAR
make analyze  # extract Item 1A -> YoY diff -> LLM scoring (cached)
make test
```

Requires [uv](https://docs.astral.sh/uv/). LLM scoring needs
`ANTHROPIC_API_KEY` in the environment (see `.env.example`); everything else
runs without a key.
