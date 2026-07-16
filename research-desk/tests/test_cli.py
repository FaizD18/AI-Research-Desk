"""Tests for CLI dispatch: each subcommand reaches its module's run_*()
with normalized tickers. The run functions are faked — nothing executes.
"""

from __future__ import annotations

import importlib

import pytest

from research_desk import cli


@pytest.mark.parametrize(
    ("command", "module_name", "attr"),
    [
        ("ingest", "research_desk.ingest", "run_ingest"),
        ("extract", "research_desk.extract", "run_extract"),
        ("diff", "research_desk.diff", "run_diff"),
        ("score", "research_desk.score", "run_score"),
        ("transcripts", "research_desk.transcripts", "run_transcripts"),
        ("thesis", "research_desk.thesis", "run_thesis"),
        ("debate", "research_desk.debate", "run_debate"),
        ("backtest", "research_desk.backtest", "run_backtest"),
    ],
)
def test_cli_dispatches_each_command(monkeypatch, command, module_name, attr) -> None:
    calls: list[list[str]] = []
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, attr, lambda tickers: calls.append(tickers))

    assert cli.main([command, "--tickers", "aapl, nvda"]) == 0
    assert calls == [["AAPL", "NVDA"]]  # normalized: upper-cased, stripped


def test_cli_rejects_unknown_command() -> None:
    with pytest.raises(SystemExit):
        cli.main(["dance"])
