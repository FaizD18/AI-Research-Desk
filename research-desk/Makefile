# AI Research Desk — all commands run through uv (never bare pip/python).
UV ?= uv

.PHONY: setup ingest extract diff score analyze debate backtest app test test-all lint fix-pth

# macOS quirk: uv can reflink the editable-install .pth with the UF_HIDDEN
# flag set, and CPython >=3.12 silently skips hidden .pth files, breaking
# `import research_desk`. Clearing the flag is a no-op elsewhere.
fix-pth:
	@find .venv -name '*.pth' -exec chflags nohidden {} + 2>/dev/null || true

setup:            ## create venv + install all deps (incl. dev)
	$(UV) sync
	@$(MAKE) --no-print-directory fix-pth

ingest:           ## fetch last 5y of 10-Ks for configured tickers into data/
	$(UV) run python -m research_desk.cli ingest

extract:          ## parse Item 1A from cached filings
	$(UV) run python -m research_desk.cli extract

diff:             ## embed + match risk paragraphs across years
	$(UV) run python -m research_desk.cli diff

score:            ## LLM-score NEW/ESCALATED risks (cached; needs ANTHROPIC_API_KEY)
	$(UV) run python -m research_desk.cli score

analyze: extract diff score  ## full Phase-1 analysis on cached filings

debate:           ## Phase 3 — Bull/Bear/Judge debate
	@echo "debate: not implemented yet (Phase 3)"

backtest:         ## Phase 3 — vectorbt long/short vs SPY
	@echo "backtest: not implemented yet (Phase 3)"

app:              ## Phase 3 — Streamlit dashboard
	@echo "app: not implemented yet (Phase 3)"

test: fix-pth     ## fast tests (network/model tests excluded)
	$(UV) run pytest

test-all: fix-pth ## everything, incl. slow tests that download the embedding model
	$(UV) run pytest -m ''

lint:
	$(UV) run ruff check src tests
