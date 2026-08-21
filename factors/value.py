# Value factor: 1/P/B (higher = cheaper = stronger value signal)

import pandas as pd


class ValueFactor:
    def __init__(self, price_data: pd.DataFrame, book_data: pd.DataFrame):
        self.price_data = price_data
        self.book_data = book_data

    def prepare_book_data(self) -> pd.DataFrame:
        """Forward-fill quarterly book value per share for daily prices."""
        book_data = self.book_data.copy()
        book_data.index = pd.to_datetime(book_data.index)
        book_data = book_data.sort_index()
        book_data = book_data.ffill()
        return book_data

    def calculate_pb(self) -> pd.DataFrame:
        """Return book yield = 1 / (Price / Book)."""
        book_data = self.prepare_book_data()
        book_data_daily = book_data.reindex(self.price_data.index, method="ffill")
        pb = self.price_data / book_data_daily
        return 1 / pb

    def calculate_value_factor(self) -> pd.DataFrame:
        return self.calculate_pb()
