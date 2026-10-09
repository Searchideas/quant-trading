"""Backtest the regime factor mix from 2017 against SPY.

Leaf weights are taken from the pre-2015 record only. A leaf keeps the
base mix when that record has fewer than 24 months. Nothing is written
into the live weight files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analytics.performance import max_drawdown, sharpe_ratio
from analytics.yield_regime import (
    BASE_WEIGHTS,
    DISCOVERY_END,
    assign_leaves,
    load_factors,
    summarise,
    _weights,
    _window_stats,
)
from backtest.engine import BacktestEngine
from strategy.benchmark import Benchmark
from strategy.factor_strategy import (
    _factor_zscore,
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


def fred_table(path: Path) -> pd.Series:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) < 2 or not parts[0][:4].isdigit():
            continue
        rows.append((pd.Timestamp(parts[0][:10]), float(parts[1])))
    series = pd.Series(dict(rows)).sort_index()
    series.index = series.index.to_period("M").to_timestamp()
    return series


def load_states_from_pages() -> pd.DataFrame:
    folder = Path(
        r"C:\Users\taiji\.cursor\projects\c-Users-taiji-Desktop-Python-Project-Quant-Learning-Journey-quant-trading\agent-tools"
    )
    gs2 = fred_table(folder / "6974c9a5-2dff-4e52-97a1-f1d94db280ef.txt")
    gs10 = fred_table(folder / "63012905-4f38-4b56-8fa0-d898700115a6.txt")
    baa = fred_table(folder / "b66e3fad-4e6c-42f2-8fa5-10c017b00378.txt")
    aaa = fred_table(folder / "7ee66eb6-5a30-4d4f-b665-426f6d350055.txt")
    return pd.DataFrame({"yield_2y": gs2, "yield_10y": gs10, "credit": baa - aaa}).dropna()


def discovery_weights(panel: pd.DataFrame) -> dict[tuple[bool, bool, bool], dict[str, float]]:
    mapping = {}
    for key in ORDER:
        block = panel[
            (panel["high_2y"] == key[0])
            & (panel["high_yield"] == key[1])
            & (panel["wide_credit"] == key[2])
        ]
        early = block[block.index <= DISCOVERY_END]
        if len(early) < 24:
            mapping[key] = dict(BASE_WEIGHTS)
            continue
        stats = _window_stats(early)
        mapping[key] = _weights(stats["winner"], stats["donor"])
    return mapping


def monthly_factor_weights(panel: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    rows = []
    for stamp, row in panel.iterrows():
        key = (bool(row["high_2y"]), bool(row["high_yield"]), bool(row["wide_credit"]))
        weights = mapping[key]
        rows.append(
            {
                "date": pd.Timestamp(stamp),
                "value": weights["value"],
                "dividend": weights["dividend"],
                "low_volatility": weights["low_volatility"],
            }
        )
    frame = pd.DataFrame(rows).set_index("date").sort_index()
    frame.index = frame.index.to_period("M").to_timestamp()
    return frame


def blend_scores(normalized: pd.DataFrame, tickers: list[str], daily_weights: pd.DataFrame) -> pd.DataFrame:
    score = pd.DataFrame(0.0, index=normalized.index, columns=tickers)
    for ticker in tickers:
        score[ticker] = (
            daily_weights["value"] * _factor_zscore(normalized, ticker, "value")
            + daily_weights["dividend"] * _factor_zscore(normalized, ticker, "dividend")
            + daily_weights["low_volatility"] * _factor_zscore(normalized, ticker, "volatility")
        )
    return score.fillna(0.0)


def run_book(prices: pd.DataFrame, stock_weights: pd.DataFrame) -> pd.Series:
    result = BacktestEngine(prices, stock_weights, initial_capital=100.0).run()
    equity = result["equity"]
    window = equity[equity.index >= "2017-01-01"]
    return 100.0 * window / float(window.iloc[0])


def month_end(series: pd.Series) -> pd.Series:
    return series.resample("ME").last().dropna()


def main() -> None:
    states = load_states_from_pages()
    panel = load_factors().join(assign_leaves(states), how="inner")
    mapping = discovery_weights(panel)
    factor_weights = monthly_factor_weights(panel, mapping)

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
    periods = pd.Series(normalized.index, index=normalized.index).dt.to_period("M").dt.to_timestamp()
    daily = factor_weights.reindex(periods.values).set_axis(normalized.index)
    daily = daily.fillna({"value": 0.60, "dividend": 0.30, "low_volatility": 0.10})

    timed_score = blend_scores(normalized, tickers, daily)
    fixed_daily = pd.DataFrame(
        {"value": 0.60, "dividend": 0.30, "low_volatility": 0.10},
        index=normalized.index,
    )
    fixed_score = blend_scores(normalized, tickers, fixed_daily)

    categorizations = data["categorizations"]
    sector_cap = data["sector_market_cap"]
    timed_book = select_sector_best_weights(timed_score, categorizations, sector_cap, max_sector_weight=1.0, top_n_per_sector=2)
    fixed_book = select_sector_best_weights(fixed_score, categorizations, sector_cap, max_sector_weight=1.0, top_n_per_sector=2)

    prices = data["price_data"]
    timed_equity = run_book(prices, timed_book)
    fixed_equity = run_book(prices, fixed_book)

    start = timed_equity.index[0].strftime("%Y-%m-%d")
    end = (timed_equity.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    spy = Benchmark("SPY", start_date=start, end_date=end, initial_capital=100.0)
    spy_equity = spy.equity.reindex(timed_equity.index).ffill().dropna()
    common = timed_equity.index.intersection(spy_equity.index)
    timed_equity = 100.0 * timed_equity.loc[common] / float(timed_equity.loc[common].iloc[0])
    fixed_equity = 100.0 * fixed_equity.reindex(common).ffill()
    fixed_equity = 100.0 * fixed_equity / float(fixed_equity.iloc[0])
    spy_equity = 100.0 * spy_equity.loc[common] / float(spy_equity.loc[common].iloc[0])

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

    yields = states.loc["2016-01-01":].copy()
    weights_from_2017 = factor_weights.loc["2017-01-01":]
    payload = {
        "summary": summarise(panel),
        "leaf_weights": {str(key): value for key, value in mapping.items()},
        "yields": [
            {
                "date": stamp.strftime("%Y-%m"),
                "yield_2y": round(float(row.yield_2y), 2),
                "yield_10y": round(float(row.yield_10y), 2),
                "credit": round(float(row.credit), 2),
            }
            for stamp, row in yields.iterrows()
        ],
        "weights": [
            {
                "date": stamp.strftime("%Y-%m"),
                "value": round(float(row.value), 2),
                "dividend": round(float(row.dividend), 2),
                "low_volatility": round(float(row.low_volatility), 2),
            }
            for stamp, row in weights_from_2017.iterrows()
        ],
        "timed": pack(timed_equity),
        "fixed": pack(fixed_equity),
        "spy": pack(spy_equity),
    }
    out = ROOT / "data" / "us" / "processed" / "regime_presentation.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(
        "timed",
        round(payload["timed"]["total_return"], 4),
        "fixed",
        round(payload["fixed"]["total_return"], 4),
        "spy",
        round(payload["spy"]["total_return"], 4),
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
