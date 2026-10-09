import pandas as pd

from factors.fundamentals import (
    ANNUAL_FILING_LAG_DAYS,
    annual_return_on_equity,
    earnings_per_share,
    growth_and_quality,
    year_ago_growth,
)


def _frame(values, dates, ticker="AAA"):
    return pd.DataFrame({ticker: values}, index=pd.to_datetime(dates))


def test_eps_is_net_income_divided_by_shares():
    income = _frame([100.0, 110.0], ["2023-12-31", "2024-12-31"])
    shares = _frame([10.0, 10.0], ["2023-12-31", "2024-12-31"])
    eps = earnings_per_share(income, shares)
    assert eps.loc["2024-12-31", "AAA"] == 11.0


def test_growth_is_eps_change_versus_prior_year():
    eps = _frame([10.0, 12.0], ["2023-12-31", "2024-12-31"])
    growth = year_ago_growth(eps)
    assert growth.loc["2024-12-31", "AAA"] == 0.2
    assert pd.isna(growth.loc["2023-12-31", "AAA"])


def test_growth_stays_blank_until_the_filing_lag():
    income = _frame([100.0, 120.0], ["2023-12-31", "2024-12-31"])
    shares = _frame([10.0, 10.0], ["2023-12-31", "2024-12-31"])
    equity = _frame([200.0, 200.0], ["2023-12-31", "2024-12-31"])
    empty = pd.DataFrame()
    dates = pd.date_range("2024-12-31", "2025-03-15", freq="D")
    growth, _ = growth_and_quality(empty, empty, empty, income, shares, equity, dates)
    published = pd.Timestamp("2024-12-31") + pd.Timedelta(days=ANNUAL_FILING_LAG_DAYS)
    assert pd.isna(growth.loc[published - pd.Timedelta(days=1), "AAA"])
    assert growth.loc[published, "AAA"] == 0.2


def test_quality_is_return_on_equity():
    income = _frame([50.0], ["2024-12-31"])
    equity = _frame([200.0], ["2024-12-31"])
    roe = annual_return_on_equity(income, equity)
    assert roe.loc["2024-12-31", "AAA"] == 0.25
