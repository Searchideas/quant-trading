# Dividend factor: higher dividend yield = stronger dividend signal

import pandas as pd


class DividendFactor:
    def __init__(self, price_data: pd.DataFrame, dividend_data: pd.DataFrame):
        self.price_data = price_data
        self.dividend_data = dividend_data

    def prepare_dividend_data(self) -> pd.DataFrame:
        """Forward-fill last dividend payment onto daily price dates."""
        div = self.dividend_data.copy()
        div.index = pd.to_datetime(div.index,utc=True)
        div.index = div.index.tz_convert(None).normalize()
        div = div.sort_index()
        return div.reindex(self.price_data.index, method="ffill")

    def calculate_dividend_yield(self) -> pd.DataFrame:
        div_daily = self.prepare_dividend_data()
        common = div_daily.columns.intersection(self.price_data.columns)
        return div_daily[common] / self.price_data[common]

    def calculate_dividend_factor(self) -> pd.DataFrame:
        return self.calculate_dividend_yield()
