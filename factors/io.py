"""Shared CSV loaders for factor inputs."""

from __future__ import annotations

import pandas as pd

from config import normalize_market
from paths import get_paths


def load_price_data(market: str | None = None) -> pd.DataFrame:
    raw_dir, _ = get_paths(market)
    prices = pd.read_csv(raw_dir / "price_data.csv", index_col=0, parse_dates=True)
    prices.index.name = "Date"
    return prices


def load_book_data(market: str | None = None) -> pd.DataFrame:
    raw_dir, _ = get_paths(market)
    book = pd.read_csv(raw_dir / "book_data.csv", index_col=0, parse_dates=True)
    book.index.name = "Date"
    return book


def load_categorizations(market: str | None = None) -> pd.DataFrame:
    raw_dir, _ = get_paths(market)
    return pd.read_csv(raw_dir / "categorizations_data.csv", index_col=0)

def load_dividend_data(market: str | None = None) -> pd.DataFrame:
    raw_dir, _ = get_paths(market)
    return pd.read_csv(raw_dir / "dividend_data.csv", index_col=0, parse_dates=True)
