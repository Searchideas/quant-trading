"""Coarse rate rule: 10-year yield and credit only.

Value and dividend may exchange 10 percentage points. Low volatility stays
at 10 percent and momentum stays at 0. A leaf moves only when it has at
least 24 months before the test starts. The live weight files are not touched.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

from analytics.performance import max_drawdown, sharpe_ratio
from analytics.regime_backtest import load_states_from_pages, month_end
from analytics.yield_regime import assign_leaves, load_factors
from backtest.engine import BacktestEngine
from strategy.benchmark import Benchmark
from strategy.factor_strategy import (
    load_strategy_data,
    normalize_factors,
    select_sector_best_weights,
)

ROOT = Path(__file__).resolve().parents[1]
SHIFT = 0.10
FLOOR = 0.05
MIN_MONTHS = 24
BASE = {"momentum": 0.0, "value": 0.60, "low_volatility": 0.10, "dividend": 0.30}
CONFIG = {"momentum": 0.0, "value": 0.65, "low_volatility": 0.05, "dividend": 0.30}
SLOW = ("value", "dividend")
# high 10-year yield, wide credit
LEAVES = (
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


def slow_tilt(block: pd.DataFrame) -> dict[str, float]:
    weights = dict(BASE)
    if len(block) < MIN_MONTHS:
        return weights
    means = {factor: float(block[factor].mean()) for factor in SLOW}
    winner = max(SLOW, key=means.get)
    donor = min(SLOW, key=means.get)
    if winner == donor or means[winner] == means[donor]:
        return weights
    move = min(SHIFT, weights[donor] - FLOOR)
    if move <= 0:
        return weights
    weights[winner] = round(weights[winner] + move, 2)
    weights[donor] = round(weights[donor] - move, 2)
    return weights


def mapping_as_of(panel: pd.DataFrame, cutoff: pd.Timestamp) -> dict[tuple[bool, bool], dict[str, float]]:
    known = panel[panel.index <= cutoff]
    mapping = {}
    for high_yield, wide_credit in LEAVES:
        block = known[(known["high_yield"] == high_yield) & (known["wide_credit"] == wide_credit)]
        mapping[(high_yield, wide_credit)] = slow_tilt(block)
    return mapping


def daily_weights(index: pd.DatetimeIndex, leaves: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    month = pd.Series(index, index=index).dt.to_period("M").dt.to_timestamp()
    aligned = leaves.reindex(month.values)
    rows = []
    for high_yield, wide_credit in zip(aligned["high_yield"], aligned["wide_credit"]):
        if pd.isna(high_yield):
            rows.append(BASE)
        else:
            rows.append(mapping[(bool(high_yield), bool(wide_credit))])
    return pd.DataFrame(rows, index=index)


def blend(normalized: pd.DataFrame, tickers: list[str], weights: pd.DataFrame) -> pd.DataFrame:
    score = pd.DataFrame(0.0, index=normalized.index, columns=tickers)
    for ticker in tickers:
        total = 0.0
        for factor, column in (
            ("value", "value"),
            ("dividend", "dividend"),
            ("low_volatility", "volatility"),
            ("momentum", "momentum"),
        ):
            name = f"{ticker}_{column}"
            series = normalized[name].fillna(0.0) if name in normalized.columns else 0.0
            total = total + weights[factor] * series
        score[ticker] = total
    return score.fillna(0.0)


def constant_weights(index: pd.DatetimeIndex, mix: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(mix, index=index)


def run_equity(prices: pd.DataFrame, stock_weights: pd.DataFrame, start: str, capital: float) -> pd.DataFrame:
    result = BacktestEngine(prices, stock_weights, initial_capital=capital).run()
    window = result[result.index >= start]
    window = window.copy()
    window["equity"] = capital * window["equity"] / float(window["equity"].iloc[0])
    return window


def pack(result: pd.DataFrame, capital: float) -> dict:
    equity = result["equity"]
    rets = equity.pct_change().dropna()
    changed = int((result.get("turnover", pd.Series(dtype=float)) > 1e-8).sum()) if "turnover" in result else 0
    gross = float((1 + result["gross_return"].fillna(0)).prod() - 1)
    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "gross_return": gross,
        "max_drawdown": float(max_drawdown(equity).min()),
        "sharpe": sharpe_ratio(rets),
        "cost_dollars": float(result["cost_sgd"].sum()) if "cost_sgd" in result else 0.0,
        "turnover_days": changed,
        "days": int(len(result)),
        "curve": [
            {"date": stamp.strftime("%Y-%m"), "value": round(float(value), 2)}
            for stamp, value in month_end(equity).items()
        ],
    }


def book(score, data) -> pd.DataFrame:
    return select_sector_best_weights(
        score,
        data["categorizations"],
        data["sector_market_cap"],
        max_sector_weight=1.0,
        top_n_per_sector=2,
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    states = load_states_from_pages()
    leaves = assign_leaves(states)
    panel = load_factors().join(leaves, how="inner")
    known_at_2017 = mapping_as_of(panel, pd.Timestamp("2016-12-31"))
    known_at_2022 = mapping_as_of(panel, pd.Timestamp("2021-12-31"))

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
    prices = data["price_data"].reindex(normalized.index)
    capital = 100_000.0

    rate_w = daily_weights(normalized.index, leaves, known_at_2017)
    hold_w = daily_weights(normalized.index, leaves, known_at_2022)
    base_w = constant_weights(normalized.index, BASE)
    config_w = constant_weights(normalized.index, CONFIG)

    rate = run_equity(prices, book(blend(normalized, tickers, rate_w), data), "2017-01-01", capital)
    base = run_equity(prices, book(blend(normalized, tickers, base_w), data), "2017-01-01", capital)
    config = run_equity(prices, book(blend(normalized, tickers, config_w), data), "2017-01-01", capital)
    common = rate.index.intersection(base.index).intersection(config.index)
    rate, base, config = rate.loc[common], base.loc[common], config.loc[common]

    end = (common[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    spy_raw = Benchmark("SPY", start_date="2017-01-01", end_date=end, initial_capital=capital).equity
    spy_equity = spy_raw.reindex(common).ffill().dropna()
    shared = common.intersection(spy_equity.index)
    rate, base, config = rate.loc[shared], base.loc[shared], config.loc[shared]

    def rebase(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["equity"] = capital * out["equity"] / float(out["equity"].iloc[0])
        return out

    rate, base, config = rebase(rate), rebase(base), rebase(config)
    spy = pd.DataFrame(
        {
            "equity": capital * spy_equity.loc[shared] / float(spy_equity.loc[shared].iloc[0]),
            "gross_return": spy_equity.loc[shared].pct_change().fillna(0.0),
            "cost_sgd": 0.0,
            "turnover": 0.0,
        }
    )

    hold = run_equity(prices, book(blend(normalized, tickers, hold_w), data), "2022-01-01", capital)
    hold_index = hold.index.intersection(spy.index[spy.index >= "2022-01-01"])
    hold = hold.loc[hold_index]
    hold["equity"] = capital * hold["equity"] / float(hold["equity"].iloc[0])
    config_hold = config.loc[hold_index].copy()
    config_hold["equity"] = capital * config_hold["equity"] / float(config_hold["equity"].iloc[0])
    spy_hold = spy.loc[hold_index].copy()
    spy_hold["equity"] = capital * spy_hold["equity"] / float(spy_hold["equity"].iloc[0])

    def describe(mapping: dict) -> list[dict]:
        rows = []
        for key in LEAVES:
            mix = mapping[key]
            rows.append(
                {
                    "name": NAMES[key],
                    "high_yield": key[0],
                    "wide_credit": key[1],
                    "momentum": mix["momentum"],
                    "value": mix["value"],
                    "low_volatility": mix["low_volatility"],
                    "dividend": mix["dividend"],
                }
            )
        return rows

    payload = {
        "capital": capital,
        "rule": "10-year yield and credit. Value and dividend exchange 10 points. Low volatility stays at 10. Momentum stays at 0.",
        "weights_known_at_2017": describe(known_at_2017),
        "weights_known_at_2022": describe(known_at_2022),
        "rate": pack(rate, capital),
        "base": pack(base, capital),
        "config": pack(config, capital),
        "spy": pack(spy, capital),
        "holdout_rate": pack(hold, capital),
        "holdout_config": pack(config_hold, capital),
        "holdout_spy": pack(spy_hold, capital),
    }
    out = ROOT / "data" / "us" / "processed" / "coarse_regime.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    print("2017-2026 rate", round(payload["rate"]["total_return"], 3), "base", round(payload["base"]["total_return"], 3), "config", round(payload["config"]["total_return"], 3), "spy", round(payload["spy"]["total_return"], 3))
    print("rate gross", round(payload["rate"]["gross_return"], 3), "cost", round(payload["rate"]["cost_dollars"], 1), "turnover days", payload["rate"]["turnover_days"])
    print("2022-2026 rate", round(payload["holdout_rate"]["total_return"], 3), "config", round(payload["holdout_config"]["total_return"], 3), "spy", round(payload["holdout_spy"]["total_return"], 3))
    for row in payload["weights_known_at_2017"]:
        print(row["name"], row["value"], row["dividend"], row["low_volatility"])
    print("wrote", out)


if __name__ == "__main__":
    main()
