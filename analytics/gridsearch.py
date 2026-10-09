"""Grid search factor weights and names per sector against SPY."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from analytics.performance import adjusted_sharpe, month_lead, sharpe_ratio
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


def _against_benchmark(strategy: pd.Series, benchmark: pd.Series) -> dict[str, float | bool]:
    """Four pass rules, all measured on the same dates as the benchmark."""
    paired = pd.concat([strategy, benchmark], axis=1, keys=["strategy", "benchmark"]).dropna()
    strat = paired["strategy"]
    bench = paired["benchmark"]
    total_return = float((1 + strat).prod() - 1)
    bench_return = float((1 + bench).prod() - 1)
    sharpe = sharpe_ratio(strat)
    bench_sharpe = sharpe_ratio(bench)
    win_rate = float((strat > 0).mean())
    bench_win_rate = float((bench > 0).mean())
    months_ahead, median_month = month_lead(strat, bench)
    return_beats = total_return > bench_return
    sharpe_beats = sharpe > bench_sharpe
    win_rate_beats = win_rate > bench_win_rate
    months_beats = months_ahead > 0.5
    return {
        "total_return": total_return,
        "benchmark_return": bench_return,
        "return_beats": return_beats,
        "sharpe": sharpe,
        "benchmark_sharpe": bench_sharpe,
        "sharpe_beats": sharpe_beats,
        "win_rate": win_rate,
        "benchmark_win_rate": bench_win_rate,
        "win_rate_beats": win_rate_beats,
        "months_ahead": months_ahead,
        "median_month_excess": median_month,
        "months_beats": months_beats,
        "passes_all": return_beats and sharpe_beats and win_rate_beats and months_beats,
    }


def gridsearch_factor_weights(
    market: str | None = None,
    step: float = 0.1,
    max_sector_weight: float | None = None,
    top_n_per_sector: int | None = None,
    top_n_values: list[int] | None = None,
    initial_capital: float | None = None,
) -> pd.DataFrame:
    """Try factor weights and 1–3 names per sector. Rank rows that beat SPY on all four tests."""
    from strategy.benchmark import Benchmark

    max_sector_weight = (
        config.MAX_SECTOR_WEIGHT if max_sector_weight is None else max_sector_weight
    )
    if top_n_values is None:
        top_n_values = [1, 2, 3] if top_n_per_sector is None else [top_n_per_sector]
    initial_capital = (
        config.INITIAL_CAPITAL if initial_capital is None else initial_capital
    )
    _, processed_dir = ensure_market_dirs(market)
    market_key = config.normalize_market(market)
    benchmark_symbol = config.get_market_config(market_key).benchmark_symbol

    data = load_strategy_data(market)
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
    prices = data["price_data"]
    start = prices.index[0].strftime("%Y-%m-%d")
    end = (prices.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    benchmark = Benchmark(
        benchmark_symbol,
        start_date=start,
        end_date=end,
        initial_capital=initial_capital,
    ).returns

    rows = []
    grid = _weight_grid(step)
    total = len(grid) * len(top_n_values)
    print(
        f"Running grid search over {len(grid)} weight combinations "
        f"and {len(top_n_values)} names-per-sector settings ({total} backtests)...",
        flush=True,
    )

    done = 0
    for top_n in top_n_values:
        for w_m, w_v, w_vol, w_div in grid:
            final_score = combine_final_score(
                normalized, tickers, w_m, w_v, w_vol, w_div
            )
            weights = select_sector_best_weights(
                final_score,
                data["categorizations"],
                data["sector_market_cap"],
                max_sector_weight=max_sector_weight,
                top_n_per_sector=top_n,
            )
            bt = BacktestEngine(
                prices,
                weights,
                initial_capital=initial_capital,
            ).run()
            rets = bt["return"].dropna()
            versus = _against_benchmark(rets, benchmark)
            rows.append(
                {
                    "top_n_per_sector": top_n,
                    "momentum_weight": w_m,
                    "value_weight": w_v,
                    "volatility_weight": w_vol,
                    "dividend_weight": w_div,
                    "adjusted_sharpe": adjusted_sharpe(rets),
                    "max_drawdown": (bt["equity"] / bt["equity"].cummax() - 1).min(),
                    "vol": rets.std() * np.sqrt(252),
                    **versus,
                }
            )
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  {done}/{total} done", flush=True)

    results = (
        pd.DataFrame(rows)
        .sort_values(["passes_all", "adjusted_sharpe"], ascending=[False, False])
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
    passed = results[results["passes_all"]]
    print(
        f"\n=== Passed all four SPY tests: {len(passed)} of {len(results)} ==="
    )
    print(
        "Tests: higher total return, higher Sharpe, higher win rate, "
        "ahead in more than half of months."
    )
    show = passed if not passed.empty else results
    best = show.iloc[0]
    print(
        f"top_n={int(best['top_n_per_sector'])}, "
        f"momentum={best['momentum_weight']:.2f}, "
        f"value={best['value_weight']:.2f}, "
        f"volatility={best['volatility_weight']:.2f}, "
        f"dividend={best['dividend_weight']:.2f}"
    )
    print(
        f"return={best['total_return']:+.2%} vs {best['benchmark_return']:+.2%}, "
        f"sharpe={best['sharpe']:.3f} vs {best['benchmark_sharpe']:.3f}, "
        f"win_rate={best['win_rate']:.1%} vs {best['benchmark_win_rate']:.1%}"
    )
    print(
        f"months ahead={best['months_ahead']:.1%}, "
        f"median month excess={best['median_month_excess']:+.2%}"
    )
    columns = [
        "top_n_per_sector",
        "momentum_weight",
        "value_weight",
        "volatility_weight",
        "dividend_weight",
        "total_return",
        "sharpe",
        "win_rate",
        "months_ahead",
        "median_month_excess",
        "passes_all",
    ]
    print(f"\nTop 10:\n{show.head(10)[columns].to_string(index=False)}")
