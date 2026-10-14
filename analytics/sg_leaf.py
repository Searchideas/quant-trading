"""Singapore yield tree for the Straits Times Index book.

The 10-year yield is the MAS SGS benchmark. Credit is the trailing
distribution yield of the iShares USD Asia high-yield ETF minus the
Amova (formerly Nikko) SGD investment-grade ETF. A month is high or wide
when its value is above the median of the previous months, and the book
uses the previous month's answers. The factor mix on each leaf is chosen
from the Singapore history, not copied from the US page.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LEAF_NAME = {
    (False, False): "Low 10-year, tight credit",
    (False, True): "Low 10-year, wide credit",
    (True, False): "High 10-year, tight credit",
    (True, True): "High 10-year, wide credit",
}
HY_TICKER = "O9P.SI"
IG_TICKER = "MBH.SI"
MEDIAN_WINDOW = 120
MEDIAN_MIN = 36

_MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def parse_sgs_10y(text: str) -> pd.Series:
    """Monthly 10-year yield from the MAS benchmark download."""
    year = None
    rows: list[tuple[pd.Timestamp, float]] = []
    for line in text.splitlines():
        parts = [part.strip().strip('"') for part in line.split(",")]
        if len(parts) < 3:
            continue
        if parts[0].isdigit() and len(parts[0]) == 4:
            year = int(parts[0])
        month = _MONTHS.get(parts[1])
        if year is None or month is None or parts[2] == "":
            continue
        try:
            value = float(parts[2])
        except ValueError:
            continue
        rows.append((pd.Timestamp(year=year, month=month, day=1), value))
    series = pd.Series({day: value for day, value in rows}, dtype=float).sort_index()
    series.index.name = "month"
    series.name = "sgs_10y"
    return series


def tiger_bond_prices(path: Path) -> pd.DataFrame:
    """Daily closes from a Tiger bar file whose symbols already end in .SI."""
    bars = pd.read_csv(path)
    dates = pd.to_datetime(bars["time"], unit="ms", utc=True).dt.tz_convert("Asia/Singapore")
    bars = bars.assign(date=dates.dt.normalize().dt.tz_localize(None))
    wide = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    wide.index.name = "Date"
    wide.columns.name = None
    return wide.sort_index()


def distribution_yield(prices: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """Trailing year of dividends divided by price, in percent, at month-end."""
    columns = {}
    for ticker in prices.columns:
        price = prices[ticker].dropna().sort_index()
        paid = dividends[ticker].dropna() if ticker in dividends.columns else pd.Series(dtype=float)
        paid = paid.sort_index()
        paid = paid.groupby(paid.index).sum()
        daily = paid.reindex(price.index).fillna(0.0)
        trailing = daily.rolling("365D").sum()
        age = pd.Series((price.index - price.index[0]).days, index=price.index)
        columns[ticker] = (trailing / price * 100.0).where(age >= 365).resample("MS").last()
    return pd.DataFrame(columns).sort_index()


def month_facts(yield_10y: pd.Series, yields: pd.DataFrame) -> pd.DataFrame:
    """Numbers behind each leaf, dated on the month that uses them.

    The row for March holds February's 10-year yield, the two bond-fund
    yields, the spread, and the medians those figures were compared with.
    """
    frame = pd.DataFrame(
        {
            "yield_10y": yield_10y,
            "hy": yields[HY_TICKER],
            "ig": yields[IG_TICKER],
            "credit": yields[HY_TICKER] - yields[IG_TICKER],
        }
    ).dropna(how="any").sort_index()
    frame["yield_median"] = frame["yield_10y"].shift(1).rolling(MEDIAN_WINDOW, min_periods=MEDIAN_MIN).median()
    frame["credit_median"] = frame["credit"].shift(1).rolling(MEDIAN_WINDOW, min_periods=MEDIAN_MIN).median()
    frame["high_yield"] = frame["yield_10y"].gt(frame["yield_median"]).mask(frame["yield_median"].isna()).astype("boolean")
    frame["wide_credit"] = frame["credit"].gt(frame["credit_median"]).mask(frame["credit_median"].isna()).astype("boolean")
    shown = frame.shift(1).dropna(subset=["high_yield", "wide_credit"])
    shown["high_yield"] = shown["high_yield"].astype(bool)
    shown["wide_credit"] = shown["wide_credit"].astype(bool)
    return shown


def credit_spread(yields: pd.DataFrame) -> pd.Series:
    """High-yield distribution yield minus investment-grade distribution yield."""
    spread = yields[HY_TICKER] - yields[IG_TICKER]
    spread.name = "credit"
    return spread.dropna()


def assign_leaves(
    yield_10y: pd.Series,
    credit: pd.Series,
    window: int = MEDIAN_WINDOW,
    min_periods: int = MEDIAN_MIN,
) -> pd.DataFrame:
    """High yield and wide credit, known from the previous month."""
    states = pd.DataFrame({"yield_10y": yield_10y, "credit": credit}).dropna(how="any")
    states = states.sort_index()
    for column, flag in (("yield_10y", "high_yield"), ("credit", "wide_credit")):
        median = states[column].shift(1).rolling(window, min_periods=min_periods).median()
        states[flag] = states[column].gt(median).mask(median.isna()).astype("boolean")
    states[["high_yield", "wide_credit", "yield_10y", "credit"]] = states[
        ["high_yield", "wide_credit", "yield_10y", "credit"]
    ].shift(1)
    states = states.dropna(subset=["high_yield", "wide_credit"])
    states["high_yield"] = states["high_yield"].astype(bool)
    states["wide_credit"] = states["wide_credit"].astype(bool)
    return states


def leaf_key(day: pd.Timestamp, leaves: pd.DataFrame):
    """Leaf already known on this day, or None before the median exists."""
    if leaves is None or leaves.empty:
        return None
    month = pd.Timestamp(day).to_period("M").to_timestamp()
    known = leaves.loc[:month]
    if known.empty:
        return None
    row = known.iloc[-1]
    return (bool(row["high_yield"]), bool(row["wide_credit"]))


def mix_on(day: pd.Timestamp, leaves: pd.DataFrame, chosen: dict) -> tuple[float, float, float, float]:
    """Singapore mix for the leaf already known on this day."""
    return chosen[leaf_key(day, leaves)]


def leaf_on(day: pd.Timestamp, leaves: pd.DataFrame) -> str:
    key = leaf_key(day, leaves)
    if key is None:
        return "Before the yield history is long enough"
    return LEAF_NAME[key]


def mix_text(mix: tuple[float, float, float, float]) -> str:
    momentum, value, volatility, dividend = mix
    return (
        f"{int(round(momentum * 100))} / {int(round(value * 100))} / "
        f"{int(round(dividend * 100))} / {int(round(volatility * 100))}"
    )


def blended_scores(
    normalized: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    members: list[str],
    leaves: pd.DataFrame,
    chosen: dict,
) -> pd.DataFrame:
    """Daily score. The mix changes with the leaf. A missing z-score counts as zero."""
    pieces = {
        key: _blend(normalized, members, mix).reindex(sessions)
        for key, mix in chosen.items()
        if key is not None
    }
    month = pd.Series(sessions, index=sessions).dt.to_period("M").dt.to_timestamp()
    flags = leaves.reindex(month.to_numpy())
    flags.index = sessions
    flags[["high_yield", "wide_credit"]] = flags[["high_yield", "wide_credit"]].ffill()
    out = pd.DataFrame(np.nan, index=sessions, columns=members)
    for key, frame in pieces.items():
        mask = (flags["high_yield"] == key[0]) & (flags["wide_credit"] == key[1])
        out.loc[mask] = frame.loc[mask]
    missing = flags["high_yield"].isna()
    if None in chosen:
        waiting = _blend(normalized, members, chosen[None]).reindex(sessions)
        out.loc[missing] = waiting.loc[missing]
    return out.ffill()


def _blend(
    normalized: pd.DataFrame,
    members: list[str],
    mix: tuple[float, float, float, float],
) -> pd.DataFrame:
    momentum, value, volatility, dividend = mix
    weights = {
        "momentum": momentum,
        "value": value,
        "volatility": volatility,
        "dividend": dividend,
    }
    columns = {}
    for name in members:
        total = pd.Series(0.0, index=normalized.index)
        for factor, weight in weights.items():
            column = f"{name}_{factor}"
            if column in normalized.columns and weight:
                total = total + weight * normalized[column].fillna(0.0)
        columns[name] = total
    return pd.DataFrame(columns, index=normalized.index)
