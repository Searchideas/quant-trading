# Volatility factor: inverse of rolling return std (higher = less volatile)

import pandas as pd


class VolatilityFactor:
    def __init__(self, price_data: pd.DataFrame):
        self.price_data = price_data

    def calculate_volatility(self, lookback: int = 30) -> pd.DataFrame:
        prices = self.price_data.ffill().bfill()
        returns = prices.pct_change()
        volatility = returns.rolling(window=lookback).std()
        volatility = volatility.iloc[lookback - 1 :]
        return 1 / volatility
