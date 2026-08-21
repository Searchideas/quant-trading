import pandas as pd

try:
    from config import COST_RATE
except ImportError:
    COST_RATE = 0.000735


def calculate_costs(
    weightage: pd.DataFrame,
    equity: pd.Series,
    cost_rate: float = COST_RATE,
) -> pd.Series:
    """Trading cost from weight changes × previous-day equity.

    turnover = sum(|Δw|) / 2  (one-way)
    cost_sgd = turnover × equity_yesterday × cost_rate
    Default cost_rate = 7.35 bps (Tiger Broker).
    """
    delta_w = (weightage - weightage.shift(1)).fillna(0.0)
    turnover = delta_w.abs().sum(axis=1) / 2

    if len(equity) == 0:
        return pd.Series(dtype=float)

    prev_equity = equity.shift(1)
    prev_equity = prev_equity.fillna(float(equity.iloc[0]))

    return (turnover * prev_equity * cost_rate).fillna(0.0)
