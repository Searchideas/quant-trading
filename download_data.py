"""Download prices, book value, and sector data for a market."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ssl_utils import configure_ssl

configure_ssl()

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    SG_TICKERS,
    US_UNIVERSE_SIZE,
    get_market_config,
    normalize_market,
)
from paths import ensure_market_dirs


def load_universe(market: str | None = None) -> list[str]:
    cfg = get_market_config(market)
    if cfg.universe_csv is not None:
        path = cfg.universe_csv
        if not path.exists():
            raise FileNotFoundError(f"Universe file not found: {path}")
        symbols = pd.read_csv(path)["Symbol"].dropna().astype(str).tolist()
        return symbols[:US_UNIVERSE_SIZE]
    return list(SG_TICKERS)


def download_prices(symbols: list[str], start_date: str) -> pd.DataFrame:
    data = yf.download(
        symbols,
        start=start_date,
        end=pd.Timestamp.today() - pd.Timedelta(days=1),
        auto_adjust=True,
        progress=False,
    )
    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame()
    close = close.dropna(axis=1, how="all")
    close.index.name = "Date"
    return close


def download_book_data(symbols: list[str]) -> pd.DataFrame:
    frames = {}
    for symbol in symbols:
        try:
            quarterly_data = yf.Ticker(symbol).quarterly_balance_sheet
            if quarterly_data.empty:
                continue
            book_value_per_share = (
                quarterly_data.loc["Common Stock Equity"]
                / quarterly_data.loc["Ordinary Shares Number"]
            )
            frames[symbol] = book_value_per_share
        except Exception as e:
            print(f"{symbol}: {e}")
    if not frames:
        return pd.DataFrame()
    book_data = pd.DataFrame(frames)
    book_data.index = pd.to_datetime(book_data.index)
    book_data = book_data.sort_index()
    book_data.index.name = "Date"
    return book_data


def sector_categorization(symbols: list[str]) -> pd.DataFrame:
    rows = {}
    for symbol in symbols:
        try:
            info = yf.Ticker(symbol).info
            sector = info.get("sector") or "Unknown"
            market_cap = info.get("marketCap") or np.nan
            rows[symbol] = {"sector": sector, "market_cap": market_cap}
        except Exception as e:
            print(f"{symbol}: {e}")
            rows[symbol] = {"sector": "Unknown", "market_cap": np.nan}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "Ticker"
    return df


def download_all(
    market: str | None = None,
    tickers: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download prices, book value, and sector data into data/{market}/raw/."""
    market = normalize_market(market)
    cfg = get_market_config(market)
    raw_dir, _ = ensure_market_dirs(market)
    tickers = tickers or load_universe(market)

    print(f"[{market}] Downloading categorizations for {len(tickers)} symbols...")
    categorizations_data = sector_categorization(tickers)
    print(f"[{market}] Downloading price data from {cfg.start_date}...")
    price_data = download_prices(tickers, cfg.start_date)
    print(f"[{market}] Downloading book data...")
    book_data = download_book_data(tickers)

    price_data.to_csv(raw_dir / "price_data.csv")
    book_data.to_csv(raw_dir / "book_data.csv")
    categorizations_data.to_csv(raw_dir / "categorizations_data.csv")
    print(f"[{market}] Saved raw data to {raw_dir}")
    return price_data, book_data, categorizations_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download market data")
    parser.add_argument("--market", default=None, choices=["us", "sg"])
    args = parser.parse_args()
    download_all(market=args.market)
