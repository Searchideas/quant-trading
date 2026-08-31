from pathlib import Path

import pandas as pd

from analytics.performance import summarize
from analytics.plots import build_rebalance_table, plot_backtest, plot_trading_costs

ANALYTICS_DIR = Path(__file__).resolve().parent


def run_analytics(
    backtest: pd.DataFrame,
    initial_capital: float = 100_000.0,
    processed_dir: Path | None = None,
    plot_path: Path | None = None,
    costs_plot_path: Path | None = None,
    weights: pd.DataFrame | None = None,
    benchmark_equity: pd.Series | None = None,
    benchmark_label: str = "Benchmark",
) -> dict[str, pd.Series | pd.DataFrame]:
    """Summarize backtest, save charts, and optional rebalance trade table."""
    from paths import get_paths

    if processed_dir is None:
        _, processed_dir = get_paths()
    plot_path = plot_path or ANALYTICS_DIR / "backtest.png"
    costs_plot_path = costs_plot_path or ANALYTICS_DIR / "trading_costs.png"

    processed_dir.mkdir(parents=True, exist_ok=True)

    stats = summarize(backtest, initial_capital)
    stats.to_csv(processed_dir / "performance_summary.csv")

    plot_backtest(
        backtest,
        save_path=plot_path,
        benchmark_equity=benchmark_equity,
        benchmark_label=benchmark_label,
    )
    plot_trading_costs(backtest, save_path=costs_plot_path)

    rebalance_table = pd.DataFrame()
    if weights is not None:
        rebalance_table = build_rebalance_table(weights, backtest)
        rebalance_table.to_csv(processed_dir / "rebalance_trades.csv")
        print(
            f"Saved rebalance table ({len(rebalance_table)} trade days) "
            f"to {processed_dir / 'rebalance_trades.csv'}"
        )

    print("\n=== Performance Summary ===")
    for key, value in stats.items():
        if isinstance(value, float):
            if key in {"total_return", "cagr", "vol", "max_drawdown", "win_rate"}:
                print(f"{key:16s}: {value:+.2%}")
            elif key in {"sharpe", "adjusted_sharpe"}:
                print(f"{key:16s}: {value:.2f}")
            else:
                print(f"{key:16s}: {value:,.2f}")
        else:
            print(f"{key:16s}: {value}")

    if not rebalance_table.empty:
        print(f"\n=== Rebalance trades (last 5) ===\n{rebalance_table.tail(5)}")

    return {"stats": stats, "rebalance_table": rebalance_table}
