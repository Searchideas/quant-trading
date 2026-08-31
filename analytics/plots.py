from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analytics.performance import max_drawdown


def _normalize_index(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.to_datetime(index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def _align_equity(equity: pd.Series, target_index: pd.Index) -> pd.Series:
    series = equity.copy()
    series.index = _normalize_index(series.index)
    series = series[~series.index.duplicated(keep="last")]
    target = _normalize_index(target_index)
    return series.reindex(target).ffill().bfill()


def plot_backtest(
    bt: pd.DataFrame,
    save_path: Path | None = None,
    benchmark_equity: pd.Series | None = None,
    benchmark_label: str = "Benchmark",
) -> None:
    dd = max_drawdown(bt["equity"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(bt.index, bt["equity"], label="Strategy", color="C0")
    if benchmark_equity is not None and len(benchmark_equity):
        bm_eq = _align_equity(benchmark_equity, bt.index)
        axes[0].plot(
            bt.index, bm_eq.values, label=benchmark_label, color="C1", alpha=0.9
        )
        axes[0].legend(loc="upper left")
    axes[0].set_title("Portfolio value")
    axes[0].set_ylabel("SGD")
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(dd.index, dd, 0, color="red", alpha=0.35, label="Strategy")
    if benchmark_equity is not None and len(benchmark_equity):
        bm_eq = _align_equity(benchmark_equity, bt.index)
        bm_dd = max_drawdown(bm_eq.dropna())
        axes[1].plot(
            bt.index, bm_dd.values, color="C1", alpha=0.9, label=benchmark_label
        )
        axes[1].legend(loc="lower left")
    axes[1].set_title("Drawdown")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(bt.index, bt["turnover"])
    axes[2].set_title("Turnover (0 = no trade that day)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_trading_costs(bt: pd.DataFrame, save_path: Path | None = None) -> None:
    """Daily cost and cumulative trading cost charts."""
    if "cost_sgd" not in bt.columns:
        print("No cost_sgd column — skip cost chart")
        return

    cost = bt["cost_sgd"].fillna(0.0)
    cumulative = cost.cumsum()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].bar(cost.index, cost.values, width=1.0, color="steelblue", alpha=0.8)
    axes[0].set_title("Daily trading cost (SGD)")
    axes[0].set_ylabel("SGD")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(cumulative.index, cumulative.values, color="darkorange")
    axes[1].set_title("Cumulative trading cost (SGD)")
    axes[1].set_ylabel("SGD")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def build_rebalance_table(
    weights: pd.DataFrame,
    backtest: pd.DataFrame | None = None,
    min_change: float = 1e-8,
) -> pd.DataFrame:
    """Table of days that require a trade (weights changed vs previous day).

    Columns: date, turnover, cost_sgd (if available), n_buys, n_sells,
    buys (tickers), sells (tickers).
    """
    delta = weights.diff()
    trade_mask = delta.abs().sum(axis=1) > min_change
    trade_days = weights.index[trade_mask]

    rows = []
    for date in trade_days:
        day_delta = delta.loc[date]
        buys = day_delta[day_delta > min_change]
        sells = day_delta[day_delta < -min_change]

        row = {
            "date": date,
            "turnover": float(day_delta.abs().sum() / 2),
            "n_buys": int(len(buys)),
            "n_sells": int(len(sells)),
            "buys": ", ".join(buys.index.tolist()),
            "sells": ", ".join(sells.index.tolist()),
            "buy_weights": ", ".join(f"{t}:{w:.2%}" for t, w in buys.items()),
            "sell_weights": ", ".join(f"{t}:{w:.2%}" for t, w in sells.items()),
        }

        if backtest is not None and date in backtest.index:
            if "cost_sgd" in backtest.columns:
                row["cost_sgd"] = float(backtest.loc[date, "cost_sgd"])
            if "equity" in backtest.columns:
                row["equity"] = float(backtest.loc[date, "equity"])

        rows.append(row)

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.set_index("date")
    return table
