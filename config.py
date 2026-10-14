"""Central settings for the quant-trading pipeline.
I am trying to see whether this can commit to github        
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Default market: override with CLI --market or env QUANT_MARKET
MARKET = os.environ.get("QUANT_MARKET", "us").lower()

# Shared strategy knobs
INITIAL_CAPITAL = 100_000.0
MAX_SECTOR_WEIGHT = 1.0
TOP_N_PER_SECTOR = 3
MOMENTUM_WEIGHT = 0
VALUE_WEIGHT = 0.7
VOLATILITY_WEIGHT = 0.1
DIVIDEND_WEIGHT = 0.2
GROWTH_WEIGHT = 0
QUALITY_WEIGHT = 0
US_UNIVERSE_SIZE = 100

SG_TICKERS = [
    "D05.SI", "O39.SI", "Z74.SI", "U11.SI", "S63.SI", "S68.SI", "J36.SI", "C6L.SI",
    "F34.SI", "H78.SI", "BN4.SI", "C38U.SI", "U14.SI", "5E2.SI", "G13.SI", "S58.SI",
    "N2IU.SI", "C09.SI", "D01.SI", "M44U.SI", "EB5.SI", "ME8U.SI", "AJBU.SI", "VC2.SI",
    "U06.SI", "OV8.SI", "V03.SI", "J69U.SI", "T82U.SI", "K71U.SI", "U10.SI", "TQ5.SI",
    "CJL.SI", "BUOU.SI", "S59.SI", "E5H.SI", "H02.SI", "HMN.SI", "A7RU.SI", "P8Z.SI",
    "C52.SI", "AWX.SI", "AIY.SI", "C2PU.SI", "F17.SI", "H15.SI", "H22.SI", "558.SI",
    "NS8U.SI", "8Y.SI", "F99.SI", "TS0U.SI", "9A4U.SI", "8C8U.SI", "SEB.SI", "Z25.SI",
    "AP4.SI", "P40U.SI", "B61.SI", "NTDU.SI", "W05.SI", "Q5T.SI", "BEC.SI", "E28.SI",
    "UIBU.SI", "AU8U.SI", "P52.SI", "S41.SI", "S61.SI", "PCT.SI", "ADN.SI", "F9D.SI",
    "FHH.SI", "J85.SI", "T6I.SI", "RE4.SI", "544.SI", "CRPU.SI", "DCRU.SI",
]


@dataclass(frozen=True)
class MarketConfig:
    key: str
    start_date: str
    benchmark_symbol: str
    universe_csv: Path | None = None  # None → use SG_TICKERS


MARKETS: dict[str, MarketConfig] = {
    "us": MarketConfig(
        key="us",
        start_date="2017-01-01",
        benchmark_symbol="SPY",
        universe_csv=ROOT / "data" / "universes" / "US_Market.csv",
    ),
    "sg": MarketConfig(
        key="sg",
        start_date="2021-09-01",  # ~5 years of history
        benchmark_symbol="ES3.SI",
        universe_csv=None,
    ),
}


def normalize_market(market: str | None = None) -> str:
    key = (market or MARKET or "us").lower().strip()
    if key in ("singapore", "si"):
        key = "sg"
    if key not in MARKETS:
        raise ValueError(f"Unknown market {market!r}; use one of {list(MARKETS)}")
    return key


def get_market_config(market: str | None = None) -> MarketConfig:
    return MARKETS[normalize_market(market)]


def get_paths(market: str | None = None) -> tuple[Path, Path]:
    """Return (raw_dir, processed_dir) for the market."""
    key = normalize_market(market)
    base = ROOT / "data" / key
    return base / "raw", base / "processed"
