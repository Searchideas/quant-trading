"""Which factor led in each yield and credit state.

Three splits, all known before the return month: the 2-year Treasury
yield, the 10-year Treasury yield, and Moody's BAA yield minus AAA.
Each series is high or wide when it is above its previous ten-year median.

Discovery is through 2014. A leaf is kept when the same factor still
has the highest average return from 2015 onward.
"""

from __future__ import annotations

import io
import json
import ssl
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ssl_utils import windows_ca_bundle

MIN_HISTORY = 120
DISCOVERY_END = pd.Timestamp("2014-12-31")
SHIFT = 0.10
FLOOR = 0.05
BASE_WEIGHTS = {"value": 0.60, "dividend": 0.30, "low_volatility": 0.10}

FRED = {
    "dgs2": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2",
    "dgs10": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10",
    "baa": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAA",
    "aaa": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=AAA",
}
FRENCH = {
    "factors": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip",
    "dividend": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Portfolios_Formed_on_D-P_CSV.zip",
    "variance": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Portfolios_Formed_on_VAR_CSV.zip",
}

FACTORS = ("value", "dividend", "low_volatility")


def _fetch(url: str) -> bytes:
    ca = windows_ca_bundle()
    context = ssl.create_default_context(cafile=ca) if ca else ssl.create_default_context()
    request = urllib.request.Request(url, headers={"User-Agent": "quant-trading"})
    with urllib.request.urlopen(request, context=context, timeout=60) as response:
        return response.read()


def _fred_monthly(url: str) -> pd.Series:
    frame = pd.read_csv(io.BytesIO(_fetch(url)))
    frame.columns = ["date", "value"]
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"])
    monthly = frame.set_index("date")["value"].resample("ME").last()
    monthly.index = monthly.index.to_period("M").to_timestamp()
    return monthly


def _french_zip(url: str) -> str:
    payload = io.BytesIO(_fetch(url))
    archive = zipfile.ZipFile(payload)
    raw = archive.read(archive.namelist()[0])
    return raw.decode("latin1")


def _french_monthly_block(text: str, start_marker: str | None = None) -> pd.DataFrame:
    """First monthly numeric block. French files mix descriptions and annual tables."""
    import csv

    lines = text.splitlines()
    start = 0
    if start_marker:
        for i, line in enumerate(lines):
            if start_marker in line:
                start = i
                break
    header_at = None
    for i in range(start, len(lines) - 1):
        if not lines[i].lstrip().startswith(","):
            continue
        if lines[i + 1][:6].isdigit():
            header_at = i
            break
    if header_at is None:
        raise ValueError("No monthly block found")
    headers = [name.strip() for name in next(csv.reader([lines[header_at]]))]
    rows = []
    for line in lines[header_at + 1 :]:
        fields = [field.strip() for field in next(csv.reader([line]))]
        if not fields or not fields[0].isdigit() or len(fields[0]) != 6:
            break
        rows.append(fields)
    frame = pd.DataFrame(rows, columns=headers)
    date = pd.to_datetime(frame.iloc[:, 0], format="%Y%m")
    values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    values.index = date
    values = values.replace([-99.99, -999], np.nan)
    values.columns = [str(column).strip() for column in values.columns]
    return values.sort_index()


def _spread(frame: pd.DataFrame, high: str, low: str) -> pd.Series:
    missing = [name for name in (high, low) if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}; have {list(frame.columns)}")
    return frame[high] - frame[low]


def load_factors() -> pd.DataFrame:
    factors = _french_monthly_block(_french_zip(FRENCH["factors"]))
    dividend = _french_monthly_block(
        _french_zip(FRENCH["dividend"]), "Value Weight Returns -- Monthly"
    )
    variance = _french_monthly_block(
        _french_zip(FRENCH["variance"]), "Value Weighted Returns -- Monthly"
    )
    out = pd.DataFrame(
        {
            "value": factors["HML"],
            "dividend": _spread(dividend, "Hi 30", "Lo 30"),
            "low_volatility": _spread(variance, "Lo 20", "Hi 20"),
        }
    )
    return out.dropna(how="any")


def load_states() -> pd.DataFrame:
    dgs2 = _fred_monthly(FRED["dgs2"])
    dgs10 = _fred_monthly(FRED["dgs10"])
    baa = _fred_monthly(FRED["baa"])
    aaa = _fred_monthly(FRED["aaa"])
    states = pd.DataFrame(
        {"yield_2y": dgs2, "yield_10y": dgs10, "credit": baa - aaa}
    ).dropna(how="any")
    return states


def _trailing_median(series: pd.Series, window: int) -> pd.Series:
    """Median of the previous `window` months, not including the current month."""
    return series.shift(1).rolling(window, min_periods=window).median()


def assign_leaves(states: pd.DataFrame, window: int = MIN_HISTORY) -> pd.DataFrame:
    labeled = states.copy()
    flags = (
        ("yield_2y", "high_2y"),
        ("yield_10y", "high_yield"),
        ("credit", "wide_credit"),
    )
    for column, flag in flags:
        labeled[flag] = labeled[column] > _trailing_median(labeled[column], window)
    # The return month uses last month's state. All three flags are known then.
    for column in ("high_2y", "high_yield", "wide_credit", "yield_2y", "yield_10y", "credit"):
        labeled[column] = labeled[column].shift(1)
    labeled = labeled.dropna(subset=["high_2y", "high_yield", "wide_credit"])
    for flag in ("high_2y", "high_yield", "wide_credit"):
        labeled[flag] = labeled[flag].astype(bool)
    return labeled


def _leaf_name(high_2y: bool, high_yield: bool, wide_credit: bool) -> str:
    two = "High 2-year yield" if high_2y else "Low 2-year yield"
    ten = "high 10-year yield" if high_yield else "low 10-year yield"
    credit_side = "wide credit" if wide_credit else "tight credit"
    return f"{two}, {ten}, {credit_side}"


def _weights(winner: str | None, donor: str | None) -> dict[str, float]:
    weights = dict(BASE_WEIGHTS)
    if winner is None or donor is None or winner == donor:
        return weights
    move = min(SHIFT, weights[donor] - FLOOR)
    if move <= 0:
        return weights
    weights[winner] = round(weights[winner] + move, 2)
    weights[donor] = round(weights[donor] - move, 2)
    return weights


def _window_stats(block: pd.DataFrame) -> dict:
    means = {factor: float(block[factor].mean()) for factor in FACTORS}
    winner = max(FACTORS, key=means.get)
    donor = min(FACTORS, key=means.get)
    return {
        "months": int(len(block)),
        "mean_monthly_pct": means,
        "mean_annual_pct": {factor: means[factor] * 12 for factor in FACTORS},
        "winner": winner,
        "donor": donor,
    }


def summarise(panel: pd.DataFrame) -> dict:
    leaves = []
    order = (
        (False, False, False),
        (False, False, True),
        (False, True, False),
        (False, True, True),
        (True, False, False),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    )
    for high_2y, high_yield, wide_credit in order:
        block = panel[
            (panel["high_2y"] == high_2y)
            & (panel["high_yield"] == high_yield)
            & (panel["wide_credit"] == wide_credit)
        ]
        discovery = block[block.index <= DISCOVERY_END]
        confirmation = block[block.index > DISCOVERY_END]
        early = _window_stats(discovery) if not discovery.empty else None
        late = _window_stats(confirmation) if not confirmation.empty else None
        kept = (
            early is not None
            and late is not None
            and early["months"] >= 24
            and late["months"] >= 24
            and early["winner"] == late["winner"]
        )
        winner = late["winner"] if kept else None
        donor = late["donor"] if kept else None
        leaves.append(
            {
                "name": _leaf_name(high_2y, high_yield, wide_credit),
                "high_2y": high_2y,
                "high_yield": high_yield,
                "wide_credit": wide_credit,
                "discovery": early,
                "confirmation": late,
                "kept": kept,
                "weights": _weights(winner, donor),
            }
        )
    latest = panel.iloc[-1]
    current = _leaf_name(
        bool(latest["high_2y"]), bool(latest["high_yield"]), bool(latest["wide_credit"])
    )
    return {
        "latest_month": panel.index[-1].strftime("%Y-%m"),
        "latest_yield_2y": float(latest["yield_2y"]),
        "latest_yield": float(latest["yield_10y"]),
        "latest_credit": float(latest["credit"]),
        "current_leaf": current,
        "leaves": leaves,
    }


def main() -> None:
    try:
        states = load_states()
    except Exception as exc:
        raise SystemExit(
            "FRED did not return the yield files from this machine "
            f"({exc}). The last completed test is in data/us/processed/yield_regime.json."
        ) from exc
    factors = load_factors()
    panel = factors.join(assign_leaves(states), how="inner")
    result = summarise(panel)
    print(json.dumps(result, indent=2))
    out = Path(__file__).resolve().parents[1] / "data" / "us" / "processed" / "yield_regime.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
