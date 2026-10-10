# Volatility factor: inverse of rolling return std (higher = less volatile)

import pandas as pd


class VolatilityFactor:
    def __init__(self, price_data: pd.DataFrame):
        self.price_data = price_data

    def calculate_volatility(self, lookback: int = 30) -> pd.DataFrame:
        # Missing prices stay missing. Filling them backward invents a history.
        returns = self.price_data.pct_change(fill_method=None)
        volatility = returns.rolling(window=lookback, min_periods=lookback).std()
        inverse = 1 / volatility
        return inverse.replace([float("inf"), float("-inf")], pd.NA)
