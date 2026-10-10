# Momentum factor: count of positive daily returns over lookback

import pandas as pd


class MomentumFactor:
    def __init__(self, price_data: pd.DataFrame):
        self.price_data = price_data

    def calculate_momentum(self, lookback: int = 45) -> pd.DataFrame:
        returns = self.price_data.pct_change(fill_method=None)
        up = returns.gt(0).astype(float).mask(returns.isna())
        return up.rolling(window=lookback, min_periods=lookback).sum()
