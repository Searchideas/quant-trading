"""Run the full quant-trading pipeline from scratch.

Steps:
  1. download_data  — fetch prices, book value, sector/market cap
  2. factors        — compute value, momentum, volatility + sector stats
  3. strategy       — build portfolio, backtest, analytics, vs benchmark
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from config import normalize_market


def run_pipeline(
    market: str | None = None,
    skip_download: bool = False,
) -> None:
    market = normalize_market(market)

    if not skip_download:
        print("=" * 60)
        print(f"Step 1/3: Downloading data [{market}]...")
        print("=" * 60)
        from download_data import download_all

        download_all(market=market)
    else:
        print(f"Skipping download [{market}]")

    print("\n" + "=" * 60)
    print(f"Step 2/3: Computing factors [{market}]...")
    print("=" * 60)
    from factors import factor_processing, load_book_data, load_price_data

    factor_processing(
        load_price_data(market),
        load_book_data(market),
        market=market,
    )

    print("\n" + "=" * 60)
    print(f"Step 3/3: Strategy, backtest & analytics [{market}]...")
    print("=" * 60)
    from strategy import run_strategy

    run_strategy(market=market)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quant-trading full pipeline")
    parser.add_argument(
        "--market",
        default=config.MARKET,
        choices=["us", "sg"],
        help="Market universe (default from config / QUANT_MARKET)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse existing raw CSVs under data/{market}/raw/",
    )
    args = parser.parse_args()
    run_pipeline(market=args.market, skip_download=args.skip_download)


if __name__ == "__main__":
    main()
