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


def download_yahoo_prices(symbols: list[str], start_date: str) -> pd.DataFrame:
    """Yahoo closes. Secondary source: fills symbols Tiger did not return."""
    data = yf.download(
        symbols,
        start=start_date,
        # Yahoo's end date is exclusive, so today includes the prior session.
        end=pd.Timestamp.today().normalize(),
        auto_adjust=True,
        progress=False,
    )
    if data is None or data.empty:
        return pd.DataFrame()
    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame()
    close = close.dropna(axis=1, how="all").dropna(axis=0, how="all")
    close.index.name = "Date"
    return close


def download_prices(symbols: list[str], start_date: str, market: str = "us") -> pd.DataFrame:
    """Tiger closes first. Yahoo fills missing symbols and is printed as a check."""
    from broker.quotes import close_mismatches, combine_closes, download_tiger_closes

    print(f"[{market}] Requesting Tiger daily closes...")
    tiger = download_tiger_closes(symbols, start_date, market)
    print(f"[{market}] Requesting Yahoo closes for the cross-check...")
    yahoo = download_yahoo_prices(symbols, start_date)
    prices, filled = combine_closes(tiger, yahoo)
    if tiger.empty:
        print(f"[{market}] Using Yahoo prices for {prices.shape[1]} symbols.")
        return prices

    mismatches = close_mismatches(tiger, yahoo)
    print(
        f"[{market}] Tiger prices: {tiger.shape[1]} symbols. "
        f"Yahoo filled {len(filled)}. "
        f"Closes differing by more than 1%: {len(mismatches)}."
    )
    if not mismatches.empty:
        preview = mismatches.head(15).copy()
        preview["gap"] = (preview["gap"] * 100).map(lambda gap: f"{gap:.2f}%")
        print(preview.to_string(index=False))
    return prices


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

def download_yahoo_dividends(symbols: list[str]) -> pd.DataFrame:
    """Yahoo dividend payments. Secondary source for symbols Tiger missed."""
    frames = {}
    for symbol in symbols:
        try:
            dividend = yf.Ticker(symbol).dividends
            if dividend.empty:
                continue
            frames[symbol] = dividend
        except Exception as e:
            print(f"{symbol}: {e}")
    if not frames:
        return pd.DataFrame()
    dividend_data = pd.DataFrame(frames)
    dividend_data.index = pd.to_datetime(dividend_data.index)
    if getattr(dividend_data.index, "tz", None) is not None:
        dividend_data.index = dividend_data.index.tz_localize(None)
    dividend_data.index = dividend_data.index.normalize()
    dividend_data = dividend_data.sort_index()
    dividend_data.index.name = "Date"
    return dividend_data


def download_dividend_data(
    symbols: list[str],
    start_date: str,
    market: str = "us",
) -> pd.DataFrame:
    """Tiger dividend history first. Yahoo fills symbols Tiger did not return."""
    from broker.quotes import combine_closes, download_tiger_dividends

    print(f"[{market}] Requesting Tiger dividends...")
    tiger = download_tiger_dividends(symbols, start_date, market)
    print(f"[{market}] Requesting Yahoo dividends for the cross-check...")
    yahoo = download_yahoo_dividends(symbols)
    dividends, filled = combine_closes(tiger, yahoo)
    if tiger.empty:
        print(f"[{market}] Using Yahoo dividends for {dividends.shape[1]} symbols.")
        return dividends
    print(
        f"[{market}] Tiger dividends: {tiger.shape[1]} symbols. "
        f"Yahoo filled {len(filled)}."
    )
    return dividends

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

    """Download dividend data"""
    print(f"[{market}] Downloading dividend data...")
    dividend_data = download_dividend_data(tickers, cfg.start_date, market)

    print(f"[{market}] Downloading categorizations for {len(tickers)} symbols...")
    categorizations_data = sector_categorization(tickers)
    print(f"[{market}] Downloading price data from {cfg.start_date}...")
    price_data = download_prices(tickers, cfg.start_date, market)
    print(f"[{market}] Downloading book data...")
    book_data = download_book_data(tickers)

    dividend_data.to_csv(raw_dir / "dividend_data.csv")
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
