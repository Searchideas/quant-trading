import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

import config
from analytics import run_analytics
from backtest.engine import BacktestEngine
from backtest.portfolio import Portfolio
from config import get_market_config, normalize_market
from paths import ensure_market_dirs
from strategy.benchmark import Benchmark
from strategy.factor_strategy import (
    factor_strategy,
    load_strategy_data,
    select_sector_best_weights,
)

__all__ = ["run_strategy"]


def run_strategy(
    market: str | None = None,
    max_sector_weight: float | None = None,
    top_n_per_sector: int | None = None,
    momentum_weight: float | None = None,
    value_weight: float | None = None,
    volatility_weight: float | None = None,
    dividend_weight: float | None = None,
    growth_weight: float | None = None,
    quality_weight: float | None = None,
    initial_capital: float | None = None,
    benchmark_symbol: str | None = None,
) -> dict[str, pd.DataFrame]:
    market = normalize_market(market)
    mcfg = get_market_config(market)
    _, processed_dir = ensure_market_dirs(market)

    max_sector_weight = (
        config.MAX_SECTOR_WEIGHT if max_sector_weight is None else max_sector_weight
    )
    top_n_per_sector = (
        config.TOP_N_PER_SECTOR if top_n_per_sector is None else top_n_per_sector
    )
    momentum_weight = (
        config.MOMENTUM_WEIGHT if momentum_weight is None else momentum_weight
    )
    value_weight = config.VALUE_WEIGHT if value_weight is None else value_weight
    volatility_weight = (
        config.VOLATILITY_WEIGHT if volatility_weight is None else volatility_weight
    )
    dividend_weight = (
        config.DIVIDEND_WEIGHT if dividend_weight is None else dividend_weight
    )
    growth_weight = config.GROWTH_WEIGHT if growth_weight is None else growth_weight
    quality_weight = config.QUALITY_WEIGHT if quality_weight is None else quality_weight
    initial_capital = (
        config.INITIAL_CAPITAL if initial_capital is None else initial_capital
    )
    benchmark_symbol = benchmark_symbol or mcfg.benchmark_symbol

    data = load_strategy_data(market)

    normalized, final_score = factor_strategy(
        data["factors_combined"],
        data["categorizations"],
        data["sector_value"],
        data["sector_momentum"],
        data["sector_volatility"],
        data["sector_dividend"],
        data["sector_growth"],
        data["sector_quality"],
        momentum_weightage=momentum_weight,
        value_weightage=value_weight,
        volatility_weightage=volatility_weight,
        dividend_weightage=dividend_weight,
        growth_weightage=growth_weight,
        quality_weightage=quality_weight,
    )

    weights = select_sector_best_weights(
        final_score,
        data["categorizations"],
        data["sector_market_cap"],
        max_sector_weight=max_sector_weight,
        top_n_per_sector=top_n_per_sector,
    )
    portfolio = Portfolio(
        weights, sectors=data["categorizations"]["sector"], capital=initial_capital
    )

    normalized.to_csv(processed_dir / "factors_normalized.csv")
    final_score.to_csv(processed_dir / "final_score.csv")
    weights.to_csv(processed_dir / "weights_history.csv")
    portfolio.current().to_csv(processed_dir / "weights_now.csv")

    backtest = BacktestEngine(
        data["price_data"],
        weights,
        initial_capital=initial_capital,
    ).run()
    backtest.to_csv(processed_dir / "backtest_results.csv")

    start = data["price_data"].index[0].strftime("%Y-%m-%d")
    end = (data["price_data"].index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    benchmark = Benchmark(
        benchmark_symbol,
        start_date=start,
        end_date=end,
        initial_capital=initial_capital,
    )

    analytics_dir = ROOT / "analytics" / market
    analytics_out = run_analytics(
        backtest,
        initial_capital=initial_capital,
        processed_dir=processed_dir,
        plot_path=analytics_dir / "backtest.png",
        costs_plot_path=analytics_dir / "trading_costs.png",
        weights=weights,
        benchmark_equity=benchmark.equity,
        benchmark_label=benchmark_symbol,
    )
    stats = analytics_out["stats"]
    rebalance_table = analytics_out["rebalance_table"]

    comparison = benchmark.print_comparison(stats, backtest["return"])
    comparison.to_csv(processed_dir / "strategy_vs_benchmark.csv")
    print(f"\nSaved comparison to {processed_dir / 'strategy_vs_benchmark.csv'}")

    rebalance_days = (weights.diff().abs().sum(axis=1) > 0).sum()
    print(f"\n[{market}] Final score: {final_score.shape} | Weights: {weights.shape}")
    print(f"Rebalance days: {rebalance_days}")
    print(
        f"Current holdings (top {top_n_per_sector}/sector, "
        f"cap-weighted max {max_sector_weight:.0%}):\n{portfolio.current()}"
    )
    print(f"Saved outputs to {processed_dir}")
    print(f"Charts: {analytics_dir / 'backtest.png'} , {analytics_dir / 'trading_costs.png'}")

    return {
        "normalized": normalized,
        "final_score": final_score,
        "weights": weights,
        "backtest": backtest,
        "portfolio": portfolio,
        "stats": stats,
        "rebalance_table": rebalance_table,
        "benchmark": benchmark,
        "comparison": comparison,
        "market": market,
    }


if __name__ == "__main__":
    run_strategy()
