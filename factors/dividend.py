# Dividend factor: higher dividend yield = stronger dividend signal

import pandas as pd

class DividendFactor:
    def __init__(self, price_data: pd.DataFrame, dividend_data: pd.DataFrame):
        self.price_data = price_data
        self.dividend_data = dividend_data

    def calculate_dividend_yield(self) -> pd.DataFrame:
        return self.dividend_data / self.price_data
    
    def calculate_dividend_factor(self) -> pd.DataFrame:
        return self.calculate_dividend_yield()