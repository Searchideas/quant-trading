"""Best four-factor mix on each 10-year yield and credit leaf.

Momentum, value, low volatility, and dividend are searched in 10 point
steps. The 2-year yield is not a split. Live weights and config.py are
left unchanged.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.leaf_weight_search import (
    CONFIG_MIX,
    ROOT,
    best_mix_for_mask,
    factor_matrix,
    grid,
    mix_label,
    pack,
    score_frame,
    stitch_scores,
)
from analytics.regime_backtest import load_states_from_pages
from analytics.yield_regime import assign_leaves
from backtest.engine import BacktestEngine
from strategy.benchmark import Benchmark
from strategy.factor_strategy import (
    load_strategy_data,
    normalize_factors,
    select_sector_best_weights,
)

ORDER = (
    (False, False),
    (False, True),
    (True, False),
    (True, True),
)
NAMES = {
    (False, False): "Low 10-year yield, tight credit",
    (False, True): "Low 10-year yield, wide credit",
    (True, False): "High 10-year yield, tight credit",
    (True, True): "High 10-year yield, wide credit",
}


def leaf_keys(index: pd.DatetimeIndex, leaves: pd.DataFrame) -> np.ndarray:
    month = pd.Series(index, index=index).dt.to_period("M").dt.to_timestamp()
    aligned = leaves.reindex(month.values)
    keys = []
    for high_yield, wide_credit in zip(aligned["high_yield"], aligned["wide_credit"]):
        if pd.isna(high_yield):
            keys.append(None)
        else:
            keys.append((bool(high_yield), bool(wide_credit)))
    return np.array(keys, dtype=object)


def choose(mixes, library, keys, active) -> dict:
    chosen = {}
    for key in ORDER:
        mask = np.array([(item == key) and flag for item, flag in zip(keys, active)], dtype=bool)
        mix, _ = best_mix_for_mask(mixes, library, mask)
        chosen[key] = mix
    return chosen


def equity_from(prices, weights, start: str, capital: float) -> pd.Series:
    result = BacktestEngine(prices, weights, initial_capital=capital).run()
    equity = result["equity"]
    window = equity[equity.index >= start]
    return window / float(window.iloc[0])


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    states = load_states_from_pages()
    leaves = assign_leaves(states)
    data = load_strategy_data("us")
    normalized, tickers = normalize_factors(
        data["factors_combined"],
        data["categorizations"],
        data["sector_value"],
        data["sector_dividend"],
        data["sector_momentum"],
        data["sector_volatility"],
        data["sector_growth"],
        data["sector_quality"],
    )
    parts = {name: factor_matrix(normalized, tickers, name) for name in ("momentum", "value", "volatility", "dividend")}
    keys = leaf_keys(normalized.index, leaves)
    prices = data["price_data"].reindex(normalized.index)
    mixes = grid()
    cache = ROOT / "data" / "us" / "processed" / "leaf_return_library.npz"
    stored = np.load(cache)
    library = stored["library"]
    if library.shape != (len(mixes), len(normalized.index)):
        raise SystemExit(f"cached returns have shape {library.shape}, expected {(len(mixes), len(normalized.index))}")

    start = pd.Timestamp("2017-01-01")
    fit_end = pd.Timestamp("2022-01-01")
    in_sample = np.array(normalized.index >= start, dtype=bool)
    train = np.array((normalized.index >= start) & (normalized.index < fit_end), dtype=bool)
    fitted = choose(mixes, library, keys, in_sample)
    held_out = choose(mixes, library, keys, train)

    categorizations = data["categorizations"]
    sector_cap = data["sector_market_cap"]

    def book(score: pd.DataFrame) -> pd.DataFrame:
        return select_sector_best_weights(
            score, categorizations, sector_cap, max_sector_weight=1.0, top_n_per_sector=2
        )

    capital = 100_000.0
    fitted_equity = equity_from(prices, book(stitch_scores(parts, normalized.index, tickers, keys, fitted)), "2017-01-01", capital)
    config_equity = equity_from(prices, book(score_frame(parts, CONFIG_MIX, normalized.index, tickers)), "2017-01-01", capital)
    common = fitted_equity.index.intersection(config_equity.index)
    end = (common[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    spy_equity = Benchmark("SPY", start_date="2017-01-01", end_date=end, initial_capital=capital).equity
    spy_equity = spy_equity.reindex(common).ffill().dropna()
    shared = common.intersection(spy_equity.index)

    def rebase(series: pd.Series) -> pd.Series:
        window = series.reindex(shared).ffill()
        return 100.0 * window / float(window.iloc[0])

    fitted_equity = rebase(fitted_equity)
    config_equity = rebase(config_equity)
    spy_equity = rebase(spy_equity)

    held_raw = equity_from(prices, book(stitch_scores(parts, normalized.index, tickers, keys, held_out)), "2022-01-01", capital)
    hold_index = held_raw.index.intersection(spy_equity.index[spy_equity.index >= "2022-01-01"])

    def rebase_hold(series: pd.Series) -> pd.Series:
        window = series.reindex(hold_index).ffill()
        return 100.0 * window / float(window.iloc[0])

    held_equity = rebase_hold(held_raw)
    config_hold = rebase_hold(config_equity)
    spy_hold = rebase_hold(spy_equity)

    leaves_out = []
    for key in ORDER:
        sample_days = sum(1 for item, flag in zip(keys, in_sample) if flag and item == key)
        train_days = sum(1 for item, flag in zip(keys, train) if flag and item == key)
        leaves_out.append(
            {
                "name": NAMES[key],
                "sample_days": sample_days,
                "train_days": train_days,
                "fitted": mix_label(fitted[key]),
                "train": mix_label(held_out[key]),
                "fitted_weights": {
                    "momentum": fitted[key][0],
                    "value": fitted[key][1],
                    "low_volatility": fitted[key][2],
                    "dividend": fitted[key][3],
                },
                "train_weights": {
                    "momentum": held_out[key][0],
                    "value": held_out[key][1],
                    "low_volatility": held_out[key][2],
                    "dividend": held_out[key][3],
                },
            }
        )

    payload = {
        "capital": capital,
        "order": "momentum / value / low volatility / dividend",
        "leaves": leaves_out,
        "fitted": pack(fitted_equity),
        "config": pack(config_equity),
        "spy": pack(spy_equity),
        "holdout_fitted": pack(held_equity),
        "holdout_config": pack(config_hold),
        "holdout_spy": pack(spy_hold),
    }
    out = ROOT / "data" / "us" / "processed" / "four_leaf_search.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    print("fitted", round(payload["fitted"]["total_return"], 3), "config", round(payload["config"]["total_return"], 3), "spy", round(payload["spy"]["total_return"], 3))
    print("holdout", round(payload["holdout_fitted"]["total_return"], 3), "config", round(payload["holdout_config"]["total_return"], 3), "spy", round(payload["holdout_spy"]["total_return"], 3))
    for leaf in leaves_out:
        print(leaf["name"], leaf["sample_days"], "all", leaf["fitted"], "before 2022", leaf["train"])
    print("wrote", out)


if __name__ == "__main__":
    main()
