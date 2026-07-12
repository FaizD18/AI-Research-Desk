"""Monthly-rebalanced long/short backtest of debate conviction scores.

The signal surface is one Judge conviction per ticker per earnings quarter
(``debate.load_convictions``), dated by the call it was derived from. Signals
are applied strictly point-in-time: a rebalance may only use signals whose
``as_of`` date lies before the rebalance date. On the first trading day of
each month every ticker whose latest thesis is convincingly long (conviction
>= CONVICTION_LONG) gets an equal long weight, convincingly short gets an
equal short weight, and everything else is flat. Weights are 1/N of the
configured universe, so gross exposure never exceeds 100%.

Prices are adjusted closes from Yahoo Finance via yfinance, cached to
``data/cache/prices/`` so reruns are offline. The strategy runs through
vectorbt with shared cash and per-side fees, benchmarked against buy-and-hold
SPY, and the run writes an honest markdown report — including where the
signal fails — to ``data/backtest_report.md``. No conviction signals in the
DB (e.g. the LLM stages haven't run yet) logs a warning and exits cleanly.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from research_desk import config, db, debate

if TYPE_CHECKING:  # pandas is heavy; modules importing us shouldn't pay for it
    import pandas as pd

log = logging.getLogger(__name__)


def signed_conviction(direction: str, conviction: int) -> int:
    """Map a thesis direction + Judge conviction to one signed signal value."""
    if direction == "long":
        return conviction
    if direction == "short":
        return -conviction
    return 0


def target_weights(
    signals: list[tuple[str, str, str, int]],
    rebalance_dates: pd.DatetimeIndex,
    universe: list[str],
) -> pd.DataFrame:
    """Point-in-time target weights per rebalance date.

    ``signals`` rows are (ticker, as_of, direction, conviction). At each
    rebalance date the latest signal with ``as_of`` strictly before the date
    decides the ticker's weight: +1/N for a convincing long, -1/N for a
    convincing short, else 0. Tickers with no prior signal are flat.
    """
    import pandas as pd

    weight = 1.0 / len(universe)
    frame = pd.DataFrame(0.0, index=rebalance_dates, columns=list(universe))
    for ticker in universe:
        history = sorted(
            (pd.Timestamp(s[1]), signed_conviction(s[2], s[3]))
            for s in signals
            if s[0] == ticker
        )
        for when in rebalance_dates:
            latest = [sc for as_of, sc in history if as_of < when]
            if not latest:
                continue
            signed = latest[-1]
            if signed >= config.CONVICTION_LONG:
                frame.loc[when, ticker] = weight
            elif signed <= -config.CONVICTION_SHORT:
                frame.loc[when, ticker] = -weight
    return frame


def load_prices(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Daily adjusted closes for ``symbols``, cache-first (one CSV per symbol)."""
    import pandas as pd

    series: dict[str, pd.Series] = {}
    for symbol in symbols:
        path = config.PRICES_CACHE_DIR / f"{symbol}.csv"
        cached: pd.Series | None = None
        if path.exists():
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
            cached = frame["close"]
            covers = (
                not cached.empty
                and cached.index[0] <= pd.Timestamp(start) + pd.Timedelta(days=7)
                and cached.index[-1] >= pd.Timestamp(end) - pd.Timedelta(days=7)
            )
            if not covers:
                cached = None
        if cached is None:
            import yfinance as yf

            log.info("%s: fetching %s..%s prices from Yahoo Finance", symbol, start, end)
            raw = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):  # yfinance MultiIndex shape
                close = close[symbol]
            cached = close.dropna()
            cached.index = pd.DatetimeIndex(cached.index).tz_localize(None)
            path.parent.mkdir(parents=True, exist_ok=True)
            cached.rename("close").to_csv(path)
        series[symbol] = cached
    prices = pd.DataFrame(series).sort_index()
    return prices.loc[str(start) : str(end)].ffill().dropna(how="any")


def avg_turnover(weights: pd.DataFrame) -> float:
    """Average one-way turnover per rebalance (target-weight based)."""
    if len(weights) < 2:
        return 0.0
    return float(weights.diff().abs().sum(axis=1).iloc[1:].mean() / 2.0)


def _write_report(
    *,
    universe: list[str],
    start: str,
    end: str,
    n_signals: int,
    per_ticker: dict[str, int],
    strategy: dict[str, float],
    benchmark: dict[str, float],
    turnover: float,
) -> None:
    def pct(x: float) -> str:
        return f"{100 * x:.1f}%"

    lines = [
        "# Backtest report — conviction-weighted long/short",
        "",
        f"Generated {date.today().isoformat()} over {start} → {end}.",
        f"Universe: {', '.join(universe)} (equal 1/N weights). Benchmark: buy-and-hold "
        f"{config.BENCHMARK}. Monthly rebalance, {config.BACKTEST_FEES * 1e4:.0f} bps "
        "fees per side.",
        f"Signals: {n_signals} debated theses "
        f"({', '.join(f'{t}: {n}' for t, n in sorted(per_ticker.items()))}); a ticker "
        f"trades only when its latest Judge conviction is >= {config.CONVICTION_LONG} "
        "(long) or the short equivalent.",
        "",
        "| metric | strategy | " + config.BENCHMARK + " |",
        "|---|---:|---:|",
        f"| total return | {pct(strategy['total_return'])} | "
        f"{pct(benchmark['total_return'])} |",
        f"| Sharpe (annualized) | {strategy['sharpe']:.2f} | {benchmark['sharpe']:.2f} |",
        f"| max drawdown | {pct(strategy['max_drawdown'])} | "
        f"{pct(benchmark['max_drawdown'])} |",
        f"| avg monthly turnover (one-way) | {pct(turnover)} | 0.0% |",
        "",
        "## Where this signal fails (read before quoting the table)",
        "",
        f"- **{len(universe)} tickers is not a universe.** Nothing here is statistically "
        "significant; single-name idiosyncrasies dominate. This is an architecture demo, "
        "not an alpha claim.",
        "- **The LLM knows the future.** Theses and debates are generated by a model whose "
        "training data overlaps the backtest window. Prompts restrict it to the provided "
        "point-in-time evidence, but knowledge leakage through the model cannot be ruled "
        "out — treat any outperformance as suspect.",
        "- **Thresholds are untuned by design.** The conviction cutoffs and 1/N sizing were "
        "fixed before results were computed and never fitted; different values give "
        "different results.",
        "- **Survivorship bias.** The tickers were picked today, so all of them survived "
        "the window.",
        "- **Cost model is approximate.** Flat per-side fees; no borrow cost on shorts, "
        "no market impact, fills at the close.",
    ]
    config.BACKTEST_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.BACKTEST_REPORT_PATH.write_text("\n".join(lines) + "\n")


def run_backtest(tickers: list[str]) -> dict[str, float] | None:
    """Backtest conviction signals for ``tickers``; returns strategy metrics.

    Returns ``None`` (after a warning, never a crash) when there are no
    debated theses yet — the LLM stages simply haven't run.
    """
    import pandas as pd
    import vectorbt as vbt

    conn = db.connect()
    rows = debate.load_convictions(conn)
    conn.close()
    universe = [t.upper() for t in tickers]
    signals = [
        (r["ticker"], r["as_of"], r["direction"], r["conviction"])
        for r in rows
        if r["ticker"] in universe
    ]
    if not signals:
        log.warning(
            "no conviction signals in the database — run `make thesis` and "
            "`make debate` first (both need ANTHROPIC_API_KEY)"
        )
        return None

    start = min(s[1] for s in signals)
    end = datetime.now(UTC).date().isoformat()
    prices = load_prices(universe + [config.BENCHMARK], start, end)
    strat_prices = prices[universe]

    month_starts = pd.date_range(start, end, freq=config.REBALANCE_FREQ)
    # First trading day on/after each month start, deduplicated.
    rebalance_dates = pd.DatetimeIndex(
        sorted(
            {
                idx[0]
                for ms in month_starts
                if len(idx := strat_prices.index[strat_prices.index >= ms]) > 0
            }
        )
    )
    weights = target_weights(signals, rebalance_dates, universe)

    size = pd.DataFrame(float("nan"), index=strat_prices.index, columns=universe)
    size.loc[weights.index] = weights
    portfolio = vbt.Portfolio.from_orders(
        strat_prices,
        size=size,
        size_type="targetpercent",
        cash_sharing=True,
        group_by=True,
        fees=config.BACKTEST_FEES,
        freq="D",
    )
    spy = vbt.Portfolio.from_holding(prices[config.BENCHMARK], freq="D")

    strategy = {
        "total_return": float(portfolio.total_return()),
        "sharpe": float(portfolio.sharpe_ratio()),
        "max_drawdown": float(portfolio.max_drawdown()),
    }
    benchmark = {
        "total_return": float(spy.total_return()),
        "sharpe": float(spy.sharpe_ratio()),
        "max_drawdown": float(spy.max_drawdown()),
    }
    per_ticker: dict[str, int] = {}
    for s in signals:
        per_ticker[s[0]] = per_ticker.get(s[0], 0) + 1
    turnover = avg_turnover(weights)
    _write_report(
        universe=universe, start=start, end=end, n_signals=len(signals),
        per_ticker=per_ticker, strategy=strategy, benchmark=benchmark,
        turnover=turnover,
    )
    log.info(
        "backtest %s..%s: strategy %.1f%% (Sharpe %.2f) vs %s %.1f%% — report at %s",
        start, end, 100 * strategy["total_return"], strategy["sharpe"],
        config.BENCHMARK, 100 * benchmark["total_return"], config.BACKTEST_REPORT_PATH,
    )
    return strategy
