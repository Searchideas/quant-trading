# Value factor: 1/P/B (higher = cheaper = stronger value signal)

import pandas as pd

from factors.fundamentals import QUARTERLY_FILING_LAG_DAYS


class ValueFactor:
    def __init__(self, price_data: pd.DataFrame, book_data: pd.DataFrame):
        self.price_data = price_data
        self.book_data = book_data

    def prepare_book_data(self) -> pd.DataFrame:
        """Carry book value forward only after a filing lag.

        Yahoo dates the figure on the quarter-end. That report is not public
        on the quarter-end, so it stays unused for 40 days and is never
        filled backward.
        """
        book_data = self.book_data.copy()
        book_data.index = pd.to_datetime(book_data.index) + pd.Timedelta(
            days=QUARTERLY_FILING_LAG_DAYS
        )
        book_data = book_data.sort_index()
        # A later row often belongs to another company and is blank here.
        # Fill the last real figure before collapsing duplicate dates.
        book_data = book_data.ffill()
        return book_data.groupby(level=0).last()

    def calculate_pb(self) -> pd.DataFrame:
        """Return book yield = 1 / (Price / Book)."""
        book_data = self.prepare_book_data()
        book_data_daily = book_data.reindex(self.price_data.index, method="ffill")
        pb = self.price_data / book_data_daily
        return 1 / pb

    def calculate_value_factor(self) -> pd.DataFrame:
        return self.calculate_pb()
