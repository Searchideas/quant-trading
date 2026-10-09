"""Search four factor weights on each yield and credit leaf.

Momentum, value, low volatility, and dividend are tried in 10 point steps
that sum to 100. Each leaf keeps the mix with the highest compounded return
on its own days. The stitched book is then run once with trading costs.
Nothing is written into the live weight files or config.py.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.performance import max_drawdown, sharpe_ratio
from analytics.regime_backtest import load_states_from_pages, month_end
from analytics.yield_regime import assign_leaves
from backtest.engine import BacktestEngine
from strategy.benchmark import Benchmark
from strategy.factor_strategy import (
    load_strategy_data,
    normalize_factors,
    select_sector_best_weights,
)

ROOT = Path(__file__).resolve().parents[1]
ORDER = (
    (False, False, False),
    (False, False, True),
    (False, True, False),
    (False, True, True),
    (True, False, False),
    (True, False, True),
    (True, True, False),
    (True, True, True),
)
NAMES = {
    (False, False, False): "Low 2-year, low 10-year, tight credit",
    (False, False, True): "Low 2-year, low 10-year, wide credit",
    (False, True, False): "Low 2-year, high 10-year, tight credit",
    (False, True, True): "Low 2-year, high 10-year, wide credit",
    (True, False, False): "High 2-year, low 10-year, tight credit",
    (True, False, True): "High 2-year, low 10-year, wide credit",
    (True, True, False): "High 2-year, high 10-year, tight credit",
    (True, True, True): "High 2-year, high 10-year, wide credit",
}
# momentum, value, low volatility, dividend
CONFIG_MIX = (0.0, 0.65, 0.05, 0.30)
BASE_MIX = (0.0, 0.60, 0.10, 0.30)


def grid(step: int = 10) -> list[tuple[float, float, float, float]]:
    mixes = []
    for momentum in range(0, 101, step):
        for value in range(0, 101 - momentum, step):
            for volatility in range(0, 101 - momentum - value, step):
                dividend = 100 - momentum - value - volatility
                mixes.append((momentum / 100, value / 100, volatility / 100, dividend / 100))
    for extra in (CONFIG_MIX, BASE_MIX):
        if extra not in mixes:
            mixes.append(extra)
    return mixes


def factor_matrix(normalized: pd.DataFrame, tickers: list[str], factor: str) -> np.ndarray:
    columns = []
    zeros = np.zeros(len(normalized), dtype=float)
    for ticker in tickers:
        name = f"{ticker}_{factor}"
        if name in normalized.columns:
            columns.append(normalized[name].fillna(0.0).to_numpy(dtype=float))
        else:
            columns.append(zeros)
    return np.column_stack(columns)


def leaf_keys(index: pd.DatetimeIndex, leaves: pd.DataFrame) -> np.ndarray:
    month = pd.Series(index, index=index).dt.to_period("M").dt.to_timestamp()
    aligned = leaves.reindex(month.values)
    keys = []
    for high_2y, high_yield, wide_credit in zip(
        aligned["high_2y"], aligned["high_yield"], aligned["wide_credit"]
    ):
        if pd.isna(high_2y):
            keys.append(None)
        else:
            keys.append((bool(high_2y), bool(high_yield), bool(wide_credit)))
    return np.array(keys, dtype=object)


def score_frame(parts: dict[str, np.ndarray], mix: tuple[float, float, float, float], index, columns) -> pd.DataFrame:
    momentum, value, volatility, dividend = mix
    values = (
        momentum * parts["momentum"]
        + value * parts["value"]
        + volatility * parts["volatility"]
        + dividend * parts["dividend"]
    )
    return pd.DataFrame(values, index=index, columns=columns)


def gross_returns(daily_returns: pd.DataFrame, weights: pd.DataFrame) -> np.ndarray:
    held = weights.shift(1).reindex(index=daily_returns.index, columns=daily_returns.columns).fillna(0.0)
    return np.nansum(held.to_numpy(dtype=float) * daily_returns.to_numpy(dtype=float), axis=1)


def best_mix_for_mask(
    mixes: list[tuple[float, float, float, float]],
    returns: np.ndarray,
    mask: np.ndarray,
) -> tuple[tuple[float, float, float, float], float]:
    if int(mask.sum()) < 2:
        return CONFIG_MIX, float("nan")
    best_mix = CONFIG_MIX
    best_value = -np.inf
    for i, mix in enumerate(mixes):
        path = returns[i, mask]
        path = path[np.isfinite(path)]
        if path.size < 2:
            continue
        value = float(np.prod(1.0 + path) - 1.0)
        if value > best_value:
            best_value = value
            best_mix = mix
    return best_mix, best_value


def run_equity(prices: pd.DataFrame, weights: pd.DataFrame, start: str) -> pd.Series:
    result = BacktestEngine(prices, weights, initial_capital=100.0).run()
    equity = result["equity"]
    window = equity[equity.index >= start]
    return 100.0 * window / float(window.iloc[0])


def pack(equity: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "max_drawdown": float(max_drawdown(equity).min()),
        "sharpe": sharpe_ratio(rets),
        "curve": [
            {"date": stamp.strftime("%Y-%m"), "value": round(float(value), 2)}
            for stamp, value in month_end(equity).items()
        ],
    }


def mix_label(mix: tuple[float, float, float, float]) -> str:
    momentum, value, volatility, dividend = mix
    return (
        f"{momentum * 100:.0f} / {value * 100:.0f} / "
        f"{volatility * 100:.0f} / {dividend * 100:.0f}"
    )


def choose(mixes, library, keys, active) -> dict[tuple, tuple]:
    chosen = {}
    for key in ORDER:
        mask = np.array([(item == key) and flag for item, flag in zip(keys, active)], dtype=bool)
        mix, _ = best_mix_for_mask(mixes, library, mask)
        chosen[key] = mix
    return chosen


def stitch_scores(parts, index, columns, keys, chosen) -> pd.DataFrame:
    momentum = np.zeros(len(index), dtype=float)
    value = np.zeros(len(index), dtype=float)
    volatility = np.zeros(len(index), dtype=float)
    dividend = np.zeros(len(index), dtype=float)
    for i, key in enumerate(keys):
        mix = chosen.get(key, CONFIG_MIX)
        momentum[i], value[i], volatility[i], dividend[i] = mix
    values = (
        momentum[:, None] * parts["momentum"]
        + value[:, None] * parts["value"]
        + volatility[:, None] * parts["volatility"]
        + dividend[:, None] * parts["dividend"]
    )
    return pd.DataFrame(values, index=index, columns=columns)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    started = time.perf_counter()
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
    parts = {
        "momentum": factor_matrix(normalized, tickers, "momentum"),
        "value": factor_matrix(normalized, tickers, "value"),
        "volatility": factor_matrix(normalized, tickers, "volatility"),
        "dividend": factor_matrix(normalized, tickers, "dividend"),
    }
    keys = leaf_keys(normalized.index, leaves)
    prices = data["price_data"].reindex(normalized.index)
    daily_returns = prices.pct_change(fill_method=None).fillna(0.0)
    mixes = grid()
    cache = ROOT / "data" / "us" / "processed" / "leaf_return_library.npz"
    library = None
    if cache.exists():
        stored = np.load(cache)
        if stored["library"].shape == (len(mixes), len(normalized.index)):
            library = stored["library"]
            print("loaded cached mix returns", flush=True)
    categorizations = data["categorizations"]
    sector_cap = data["sector_market_cap"]

    if library is None:
        library = np.zeros((len(mixes), len(normalized.index)), dtype=float)
        for i, mix in enumerate(mixes):
            frame = score_frame(parts, mix, normalized.index, tickers)
            book = select_sector_best_weights(
                frame, categorizations, sector_cap, max_sector_weight=1.0, top_n_per_sector=2
            )
            library[i] = gross_returns(daily_returns, book)
            if i % 40 == 0:
                print(f"mix {i + 1}/{len(mixes)} {time.perf_counter() - started:.0f}s", flush=True)
        np.savez_compressed(cache, library=library)

    start = pd.Timestamp("2017-01-01")
    fit_end = pd.Timestamp("2022-01-01")
    in_sample = np.array(normalized.index >= start, dtype=bool)
    train = np.array((normalized.index >= start) & (normalized.index < fit_end), dtype=bool)
    fitted = choose(mixes, library, keys, in_sample)
    held_out = choose(mixes, library, keys, train)

    fitted_score = stitch_scores(parts, normalized.index, tickers, keys, fitted)
    held_score = stitch_scores(parts, normalized.index, tickers, keys, held_out)
    config_score = score_frame(parts, CONFIG_MIX, normalized.index, tickers)

    def book(score: pd.DataFrame) -> pd.DataFrame:
        return select_sector_best_weights(
            score, categorizations, sector_cap, max_sector_weight=1.0, top_n_per_sector=2
        )

    fitted_equity = run_equity(prices, book(fitted_score), "2017-01-01")
    config_equity = run_equity(prices, book(config_score), "2017-01-01")
    common = fitted_equity.index.intersection(config_equity.index)
    fitted_equity = 100.0 * fitted_equity.loc[common] / float(fitted_equity.loc[common].iloc[0])
    config_equity = 100.0 * config_equity.loc[common] / float(config_equity.loc[common].iloc[0])

    end = (common[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    spy = Benchmark("SPY", start_date="2017-01-01", end_date=end, initial_capital=100.0)
    spy_equity = spy.equity.reindex(common).ffill().dropna()
    shared = common.intersection(spy_equity.index)
    fitted_equity = 100.0 * fitted_equity.loc[shared] / float(fitted_equity.loc[shared].iloc[0])
    config_equity = 100.0 * config_equity.loc[shared] / float(config_equity.loc[shared].iloc[0])
    spy_equity = 100.0 * spy_equity.loc[shared] / float(spy_equity.loc[shared].iloc[0])

    held_equity = run_equity(prices, book(held_score), "2022-01-01")
    spy_hold = spy_equity[spy_equity.index >= "2022-01-01"]
    hold_index = held_equity.index.intersection(spy_hold.index)
    held_equity = 100.0 * held_equity.loc[hold_index] / float(held_equity.loc[hold_index].iloc[0])
    spy_hold = 100.0 * spy_hold.loc[hold_index] / float(spy_hold.loc[hold_index].iloc[0])
    config_hold = config_equity.reindex(hold_index).ffill()
    config_hold = 100.0 * config_hold / float(config_hold.iloc[0])

    leaves_out = []
    for key in ORDER:
        sample_days = sum(1 for item, flag in zip(keys, in_sample) if flag and item == key)
        train_days = sum(1 for item, flag in zip(keys, train) if flag and item == key)
        leaves_out.append(
            {
                "name": NAMES[key],
                "high_2y": key[0],
                "high_yield": key[1],
                "wide_credit": key[2],
                "sample_days": sample_days,
                "train_days": train_days,
                "fitted": {
                    "momentum": fitted[key][0],
                    "value": fitted[key][1],
                    "low_volatility": fitted[key][2],
                    "dividend": fitted[key][3],
                },
                "train": {
                    "momentum": held_out[key][0],
                    "value": held_out[key][1],
                    "low_volatility": held_out[key][2],
                    "dividend": held_out[key][3],
                },
            }
        )

    payload = {
        "window": "2017-01 to the latest price",
        "fit_note": "Each leaf's weights were chosen on that leaf's own days inside the reported window.",
        "holdout_note": "Holdout weights were chosen only on days before 2022, then measured from 2022.",
        "leaves": leaves_out,
        "fitted": pack(fitted_equity),
        "config": pack(config_equity),
        "spy": pack(spy_equity),
        "holdout_fitted": pack(held_equity),
        "holdout_config": pack(config_hold),
        "holdout_spy": pack(spy_hold),
    }
    out = ROOT / "data" / "us" / "processed" / "leaf_weight_search.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    print("fitted", round(payload["fitted"]["total_return"], 4), "config", round(payload["config"]["total_return"], 4), "spy", round(payload["spy"]["total_return"], 4))
    print("holdout", round(payload["holdout_fitted"]["total_return"], 4), "spy", round(payload["holdout_spy"]["total_return"], 4))
    for leaf in leaves_out:
        mix = leaf["fitted"]
        print(leaf["name"], leaf["sample_days"], mix_label((mix["momentum"], mix["value"], mix["low_volatility"], mix["dividend"])))
    print("wrote", out, f"{time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
