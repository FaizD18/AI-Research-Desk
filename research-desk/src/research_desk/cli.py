"""Command-line entry points for each pipeline stage.

Usage: python -m research_desk.cli {ingest,extract,diff,score} [--tickers AAPL,NVDA]
Each subcommand is a thin wrapper around one module's run_*() function so the
Makefile, tests, and notebooks all share the same code path.
"""

from __future__ import annotations

import argparse
import logging

from research_desk import config


def _parse_tickers(raw: str | None) -> list[str]:
    """Turn a comma-separated --tickers value into a list, defaulting to config."""
    if not raw:
        return config.TICKERS
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def main(argv: list[str] | None = None) -> int:
    """Dispatch to a pipeline stage; returns a process exit code."""
    parser = argparse.ArgumentParser(prog="research_desk", description=__doc__)
    parser.add_argument("--tickers", help="comma-separated tickers (default: config.TICKERS)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "command",
        choices=["ingest", "extract", "diff", "score", "transcripts", "thesis", "debate",
                 "backtest"],
        help="pipeline stage to run",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    tickers = _parse_tickers(args.tickers)

    if args.command == "ingest":
        from research_desk.ingest import run_ingest

        run_ingest(tickers)
    elif args.command == "extract":
        from research_desk.extract import run_extract

        run_extract(tickers)
    elif args.command == "diff":
        from research_desk.diff import run_diff

        run_diff(tickers)
    elif args.command == "score":
        from research_desk.score import run_score

        run_score(tickers)
    elif args.command == "transcripts":
        from research_desk.transcripts import run_transcripts

        run_transcripts(tickers)
    elif args.command == "thesis":
        from research_desk.thesis import run_thesis

        run_thesis(tickers)
    elif args.command == "debate":
        from research_desk.debate import run_debate

        run_debate(tickers)
    elif args.command == "backtest":
        from research_desk.backtest import run_backtest

        run_backtest(tickers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
