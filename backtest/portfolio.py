import pandas as pd


class Portfolio:
    """Target weights over time, plus the latest holdings snapshot."""

    def __init__(
        self,
        weights: pd.DataFrame,
        sectors: pd.Series | None = None,
        capital: float = 100_000.0,
    ):
        self.weights = weights.fillna(0.0)
        self.sectors = sectors
        self.capital = capital

    def historical(self) -> pd.DataFrame:
        """Daily weight for every ticker. 0 means not held that day."""
        return self.weights

    def current(self, min_names: int = 2) -> pd.DataFrame:
        """Latest holdings on a day when the book is fully invested.

        The last calendar rows can have missing prices, which collapses
        the book to one name. Use the last complete day instead.
        """
        n_names = (self.weights > 1e-12).sum(axis=1)
        complete = n_names[n_names >= min_names]
        as_of = complete.index[-1] if not complete.empty else self.weights.index[-1]
        latest = self.weights.loc[as_of]
        held = latest[latest > 1e-12].sort_values(ascending=False)
        table = pd.DataFrame(
            {
                "weight": held,
                "weight_pct": held * 100,
                "amount_sgd": held * self.capital,
            }
        )
        if self.sectors is not None:
            table["sector"] = self.sectors.reindex(table.index)
            table = table[["sector", "weight", "weight_pct", "amount_sgd"]]
        table.index.name = "ticker"
        table.attrs["as_of"] = as_of
        return table
