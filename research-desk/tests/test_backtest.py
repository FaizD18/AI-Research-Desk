"""Tests for the conviction backtest: point-in-time weights, signal mapping,
report generation, and graceful no-signal behavior. Prices are synthetic —
no yfinance, no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_desk import backtest, config, db


@pytest.fixture()
def _tmp_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "b.db")
    monkeypatch.setattr(config, "BACKTEST_REPORT_PATH", tmp_path / "report.md")
    monkeypatch.setattr(config, "PRICES_CACHE_DIR", tmp_path / "prices")
    return tmp_path


def _seed_signal(ticker: str, as_of: str, direction: str, conviction: int) -> None:
    """Insert the transcript -> thesis -> debate chain behind one signal."""
    conn = db.connect(config.DB_PATH)
    cur = conn.execute(
        """INSERT INTO transcripts (ticker, fiscal_year, fiscal_quarter, call_date,
           word_count, hedging_per_1k, uncertainty_per_1k, guidance_per_1k,
           text_path, source, ingested_at)
           VALUES (?, 2026, 1, ?, 8000, 5.0, 2.0, 4.0, 'p', 'defeatbeta', 't')""",
        (ticker, as_of),
    )
    cur = conn.execute(
        """INSERT INTO theses (ticker, transcript_id, as_of, direction, confidence,
           summary, evidence_json, model, created_at)
           VALUES (?, ?, ?, ?, 0.7, 's', '[]', 'm', 't')""",
        (ticker, cur.lastrowid, as_of, direction),
    )
    conn.execute(
        """INSERT INTO debates (thesis_id, conviction, judge_reasoning,
           transcript_path, model, debated_at)
           VALUES (?, ?, 'r', 'p', 'm', 't')""",
        (cur.lastrowid, conviction),
    )
    conn.commit()
    conn.close()


def _synthetic_prices(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Deterministic daily closes: AAPL trends up, NVDA down, SPY flat-ish."""
    idx = pd.bdate_range(start, end)
    n = len(idx)
    shapes = {
        "AAPL": np.linspace(100, 140, n),
        "NVDA": np.linspace(100, 70, n),
        "SPY": np.linspace(100, 104, n),
    }
    return pd.DataFrame({s: shapes[s] for s in symbols}, index=idx)


# --- signed_conviction / target_weights -------------------------------------


def test_signed_conviction_maps_directions() -> None:
    assert backtest.signed_conviction("long", 80) == 80
    assert backtest.signed_conviction("short", 80) == -80
    assert backtest.signed_conviction("neutral", 95) == 0


def test_target_weights_are_point_in_time() -> None:
    dates = pd.DatetimeIndex(["2026-02-02", "2026-03-02", "2026-04-01"])
    signals = [("AAPL", "2026-03-02", "long", 90)]  # known at end of 3/2
    w = backtest.target_weights(signals, dates, ["AAPL", "NVDA"])
    assert w.loc["2026-02-02", "AAPL"] == 0.0  # before the signal exists
    assert w.loc["2026-03-02", "AAPL"] == 0.0  # not strictly before the date
    assert w.loc["2026-04-01", "AAPL"] == 0.5  # 1/N with N=2


def test_target_weights_thresholds_and_shorts() -> None:
    dates = pd.DatetimeIndex(["2026-04-01"])
    signals = [
        ("AAPL", "2026-01-29", "long", 59),  # below CONVICTION_LONG -> flat
        ("NVDA", "2026-02-25", "short", 75),  # convincing short
    ]
    w = backtest.target_weights(signals, dates, ["AAPL", "NVDA"])
    assert w.loc["2026-04-01", "AAPL"] == 0.0
    assert w.loc["2026-04-01", "NVDA"] == -0.5


def test_target_weights_use_latest_signal_only() -> None:
    dates = pd.DatetimeIndex(["2026-06-01"])
    signals = [
        ("AAPL", "2026-01-29", "long", 90),
        ("AAPL", "2026-04-30", "neutral", 90),  # newer: overrides the long
    ]
    w = backtest.target_weights(signals, dates, ["AAPL"])
    assert w.loc["2026-06-01", "AAPL"] == 0.0


def test_avg_turnover_zero_for_constant_targets() -> None:
    w = pd.DataFrame({"AAPL": [0.5, 0.5, 0.5]},
                     index=pd.DatetimeIndex(["2026-01-01", "2026-02-01", "2026-03-01"]))
    assert backtest.avg_turnover(w) == 0.0


def test_avg_turnover_scales_one_way() -> None:
    # Full flip: |Δ| sums to 2.0 across both legs -> one-way turnover 1.0.
    w = pd.DataFrame(
        {"AAPL": [0.5, -0.5], "NVDA": [-0.5, 0.5]},
        index=pd.DatetimeIndex(["2026-01-01", "2026-02-01"]),
    )
    assert backtest.avg_turnover(w) == pytest.approx(1.0)


# --- load_prices -------------------------------------------------------------


def _install_yf_stub(monkeypatch, download):
    import sys
    import types

    mod = types.ModuleType("yfinance")
    mod.download = download
    monkeypatch.setitem(sys.modules, "yfinance", mod)


def test_load_prices_fetches_writes_cache_then_reads_offline(_tmp_env, monkeypatch) -> None:
    idx = pd.date_range("2026-01-02", "2026-03-31", freq="B", tz="UTC")

    def fake_download(symbol, start, end, **kwargs):
        # yfinance's modern shape: MultiIndex (field, symbol), tz-aware index.
        return pd.DataFrame(
            {("Close", symbol): np.linspace(100, 110, len(idx))}, index=idx
        )

    _install_yf_stub(monkeypatch, fake_download)
    first = backtest.load_prices(["AAPL"], "2026-01-02", "2026-03-31")
    assert list(first.columns) == ["AAPL"]
    assert (config.PRICES_CACHE_DIR / "AAPL.csv").exists()
    assert first.index.tz is None  # normalized for comparisons downstream

    def poisoned_download(*args, **kwargs):
        raise AssertionError("cache miss: network hit on a covered window")

    _install_yf_stub(monkeypatch, poisoned_download)
    second = backtest.load_prices(["AAPL"], "2026-01-02", "2026-03-31")
    assert second["AAPL"].iloc[-1] == pytest.approx(first["AAPL"].iloc[-1])


def test_load_prices_refetches_when_cache_window_too_short(_tmp_env, monkeypatch) -> None:
    calls: list[str] = []

    def fake_download(symbol, start, end, **kwargs):
        calls.append(f"{start}..{end}")
        idx = pd.date_range(start, end, freq="B")
        return pd.DataFrame({"Close": np.linspace(100, 101, len(idx))}, index=idx)

    _install_yf_stub(monkeypatch, fake_download)
    backtest.load_prices(["AAPL"], "2026-01-02", "2026-02-27")
    # Asking well past the cached end must refetch, not serve stale data.
    backtest.load_prices(["AAPL"], "2026-01-02", "2026-06-30")
    assert len(calls) == 2


# --- run_backtest ------------------------------------------------------------


def test_run_backtest_no_signals_returns_none(_tmp_env) -> None:
    db.connect(config.DB_PATH).close()  # empty schema
    assert backtest.run_backtest(["AAPL", "NVDA"]) is None
    assert not config.BACKTEST_REPORT_PATH.exists()


def test_run_backtest_end_to_end_with_synthetic_prices(_tmp_env, monkeypatch) -> None:
    # Long the up-trender, short the down-trender, from January calls.
    _seed_signal("AAPL", "2026-01-05", "long", 85)
    _seed_signal("NVDA", "2026-01-05", "short", 80)
    monkeypatch.setattr(backtest, "load_prices", _synthetic_prices)

    metrics = backtest.run_backtest(["AAPL", "NVDA"])
    assert metrics is not None
    assert metrics["total_return"] > 0  # rigged: both legs were right
    assert np.isfinite(metrics["sharpe"])
    assert np.isfinite(metrics["max_drawdown"]) and metrics["max_drawdown"] <= 0.0

    report = config.BACKTEST_REPORT_PATH.read_text()
    assert "Sharpe" in report and "SPY" in report
    assert "Where this signal fails" in report
    assert "AAPL: 1" in report and "NVDA: 1" in report


def test_run_backtest_flat_when_signals_unconvincing(_tmp_env, monkeypatch) -> None:
    _seed_signal("AAPL", "2026-01-05", "long", 50)  # below the bar
    monkeypatch.setattr(backtest, "load_prices", _synthetic_prices)
    metrics = backtest.run_backtest(["AAPL"])
    assert metrics is not None
    assert metrics["total_return"] == pytest.approx(0.0, abs=1e-9)  # never traded
