from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.costs import calculate_costs


class BacktestEngine:
    def __init__(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        initial_capital: float = 100_000.0,
    ):
        self.prices = prices
        self.weights = weights.reindex(
            index=prices.index, columns=prices.columns
        ).fillna(0.0)
        self.initial_capital = initial_capital

    def run(self) -> pd.DataFrame:
        returns = self.prices.pct_change(fill_method=None)
        held = self.weights.shift(1)  # no lookahead

        gross_return = (held * returns).sum(axis=1)

        # equity before costs (to size notional)
        equity_gross = self.initial_capital * (1 + gross_return.fillna(0.0)).cumprod()

        prev_equity = equity_gross.shift(1)
        if len(prev_equity):
            prev_equity.iloc[0] = self.initial_capital
        cost_sgd = calculate_costs(self.weights, prev_equity, self.prices)
        cost_pct = (cost_sgd / prev_equity.replace(0, pd.NA)).fillna(0.0)

        net_return = gross_return.fillna(0.0) - cost_pct
        equity = self.initial_capital * (1 + net_return).cumprod()

        turnover = (self.weights - self.weights.shift(1)).abs().sum(axis=1) / 2

        return pd.DataFrame(
            {
                "return": net_return,
                "gross_return": gross_return,
                "equity": equity,
                "cost_sgd": cost_sgd,
                "turnover": turnover,
                "n_names": (held > 0).sum(axis=1),
            }
        ).dropna(subset=["return"])