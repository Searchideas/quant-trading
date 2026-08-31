"""Grid search factor weights for best adjusted Sharpe."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from analytics.performance import adjusted_sharpe, sharpe_ratio
from backtest.engine import BacktestEngine
from paths import ensure_market_dirs
from strategy.factor_strategy import (
    combine_final_score,
    load_strategy_data,
    normalize_factors,
    select_sector_best_weights,
)


def _weight_grid(step: float = 0.1) -> list[tuple[float, float, float, float]]:
    """All (momentum, value, volatility, dividend) weights that sum to 1."""
    weights = []
    for w_m in np.arange(0, 1.0 + step / 2, step):
        for w_v in np.arange(0, 1.0 - w_m + step / 2, step):
            for w_vol in np.arange(0, 1.0 - w_m - w_v + step / 2, step):
                w_div = round(1.0 - w_m - w_v - w_vol, 4)
                if w_div < -1e-9:
                    continue
                weights.append(
                    (
                        round(w_m, 4),
                        round(w_v, 4),
                        round(max(w_vol, 0.0), 4),
                        round(max(w_div, 0.0), 4),
                    )
                )
    return weights


def gridsearch_factor_weights(
    market: str | None = None,
    step: float = 0.1,
    max_sector_weight: float | None = None,
    top_n_per_sector: int | None = None,
    initial_capital: float | None = None,
) -> pd.DataFrame:
    """Try all factor weight combos; return results sorted by adjusted Sharpe."""
    max_sector_weight = (
        config.MAX_SECTOR_WEIGHT if max_sector_weight is None else max_sector_weight
    )
    top_n_per_sector = (
        config.TOP_N_PER_SECTOR if top_n_per_sector is None else top_n_per_sector
    )
    initial_capital = (
        config.INITIAL_CAPITAL if initial_capital is None else initial_capital
    )
    _, processed_dir = ensure_market_dirs(market)

    data = load_strategy_data(market)
    normalized, tickers = normalize_factors(
        data["factors_combined"],
        data["categorizations"],
        data["sector_value"],
        data["sector_dividend"],
        data["sector_momentum"],
        data["sector_volatility"],
    )

    rows = []
    grid = _weight_grid(step)
    print(f"Running grid search over {len(grid)} weight combinations...", flush=True)

    for i, (w_m, w_v, w_vol, w_div) in enumerate(grid, 1):
        final_score = combine_final_score(
            normalized, tickers, w_m, w_v, w_vol, w_div
        )
        weights = select_sector_best_weights(
            final_score,
            data["categorizations"],
            data["sector_market_cap"],
            max_sector_weight=max_sector_weight,
            top_n_per_sector=top_n_per_sector,
        )
        bt = BacktestEngine(
            data["price_data"],
            weights,
            initial_capital=initial_capital,
            cost_rate=config.COST_RATE,
        ).run()
        rets = bt["return"].dropna()

        rows.append(
            {
                "momentum_weight": w_m,
                "value_weight": w_v,
                "volatility_weight": w_vol,
                "dividend_weight": w_div,
                "sharpe": sharpe_ratio(rets),
                "adjusted_sharpe": adjusted_sharpe(rets),
                "total_return": bt["equity"].iloc[-1] / initial_capital - 1,
                "max_drawdown": (bt["equity"] / bt["equity"].cummax() - 1).min(),
                "vol": rets.std() * np.sqrt(252),
            }
        )
        if i % 50 == 0 or i == len(grid):
            print(f"  {i}/{len(grid)} done")

    results = (
        pd.DataFrame(rows)
        .sort_values("adjusted_sharpe", ascending=False)
        .reset_index(drop=True)
    )
    out = processed_dir / "gridsearch_results.csv"
    results.to_csv(out, index=False)
    print(f"Saved full results to {out}")
    return results


if __name__ == "__main__":
    import argparse
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", default=None, choices=["us", "sg"])
    parser.add_argument("--step", type=float, default=0.1)
    args = parser.parse_args()

    results = gridsearch_factor_weights(market=args.market, step=args.step)
    best = results.iloc[0]
    print("\n=== Best by Adjusted Sharpe ===")
    print(
        f"momentum={best['momentum_weight']:.2f}, "
        f"value={best['value_weight']:.2f}, "
        f"volatility={best['volatility_weight']:.2f}, "
        f"dividend={best['dividend_weight']:.2f}"
    )
    print(
        f"adjusted_sharpe={best['adjusted_sharpe']:.3f}, sharpe={best['sharpe']:.3f}"
    )
    print(
        f"total_return={best['total_return']:+.2%}, "
        f"max_drawdown={best['max_drawdown']:+.2%}"
    )
    print(f"\nTop 10 combinations:\n{results.head(10).to_string(index=False)}")
