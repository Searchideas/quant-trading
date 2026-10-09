"""Pick the highest-Sharpe four-factor mix on each 10-year and credit leaf.

The search uses the cached daily returns of every 10-point mix. Live
weights and config.py are not changed.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from analytics.four_leaf_search import NAMES, ORDER, leaf_keys
from analytics.leaf_weight_search import ROOT, grid
from analytics.regime_backtest import load_states_from_pages
from analytics.yield_regime import assign_leaves
from strategy.factor_strategy import load_strategy_data, normalize_factors


def best_sharpe(mixes, library, mask) -> tuple[tuple[float, float, float, float], float]:
    best_mix = mixes[0]
    best_value = -np.inf
    for i, mix in enumerate(mixes):
        path = library[i, mask]
        path = path[np.isfinite(path)]
        if path.size < 20 or float(path.std()) == 0:
            continue
        value = float(path.mean() / path.std() * np.sqrt(252))
        if value > best_value:
            best_value = value
            best_mix = mix
    return best_mix, best_value


def spells(months: pd.DatetimeIndex, keys: list) -> list[dict]:
    rows = []
    start = months[0]
    current = keys[0]
    previous = months[0]
    for stamp, key in zip(months[1:], keys[1:]):
        if key != current:
            rows.append({"start": start.strftime("%Y-%m"), "end": previous.strftime("%Y-%m"), "key": current})
            start = stamp
            current = key
        previous = stamp
    rows.append({"start": start.strftime("%Y-%m"), "end": previous.strftime("%Y-%m"), "key": current})
    return rows


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    leaves = assign_leaves(load_states_from_pages())
    data = load_strategy_data("us")
    normalized, _tickers = normalize_factors(
        data["factors_combined"],
        data["categorizations"],
        data["sector_value"],
        data["sector_dividend"],
        data["sector_momentum"],
        data["sector_volatility"],
        data["sector_growth"],
        data["sector_quality"],
    )
    keys = leaf_keys(normalized.index, leaves)
    mixes = grid()
    library = np.load(ROOT / "data" / "us" / "processed" / "leaf_return_library.npz")["library"]
    if library.shape != (len(mixes), len(normalized.index)):
        raise SystemExit(f"cached returns have shape {library.shape}")

    active = np.array(normalized.index >= "2017-01-01", dtype=bool)
    chosen = {}
    leaf_rows = []
    for key in ORDER:
        mask = np.array([(item == key) and flag for item, flag in zip(keys, active)], dtype=bool)
        mix, sharpe = best_sharpe(mixes, library, mask)
        chosen[key] = mix
        leaf_rows.append(
            {
                "name": NAMES[key],
                "high_yield": key[0],
                "wide_credit": key[1],
                "days": int(mask.sum()),
                "sharpe": round(sharpe, 2),
                "momentum": mix[0],
                "value": mix[1],
                "low_volatility": mix[2],
                "dividend": mix[3],
            }
        )

    month_index = pd.date_range("2017-01-01", leaves.index.max(), freq="MS")
    aligned = leaves.reindex(month_index)
    month_keys = []
    weights = []
    for stamp, row in aligned.iterrows():
        if pd.isna(row.high_yield):
            continue
        key = (bool(row.high_yield), bool(row.wide_credit))
        mix = chosen[key]
        month_keys.append(key)
        weights.append(
            {
                "date": pd.Timestamp(stamp).strftime("%Y-%m"),
                "leaf": NAMES[key],
                "momentum": round(mix[0] * 100, 1),
                "value": round(mix[1] * 100, 1),
                "dividend": round(mix[3] * 100, 1),
                "low_volatility": round(mix[2] * 100, 1),
            }
        )
    periods = []
    for spell in spells(pd.to_datetime([row["date"] for row in weights]), [row["leaf"] for row in weights]):
        periods.append(
            {
                "start": spell["start"],
                "end": spell["end"],
                "leaf": spell["key"],
            }
        )

    payload = {
        "order": "momentum / value / dividend / low volatility",
        "leaves": leaf_rows,
        "periods": periods,
        "weights": weights,
        "latest": weights[-1],
    }
    out = ROOT / "data" / "us" / "processed" / "four_leaf_sharpe.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    for row in leaf_rows:
        print(
            row["name"],
            row["days"],
            f"{row['momentum']*100:.0f}/{row['value']*100:.0f}/{row['dividend']*100:.0f}/{row['low_volatility']*100:.0f}",
            "sharpe",
            row["sharpe"],
        )
    print("periods", len(periods), "latest", weights[-1]["date"], weights[-1]["leaf"])
    print("wrote", out)


if __name__ == "__main__":
    main()
