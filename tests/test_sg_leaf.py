import pandas as pd

from analytics.sg_leaf import HY_TICKER, IG_TICKER, assign_leaves, credit_spread, month_facts, parse_sgs_10y


def test_mas_year_is_carried_onto_the_following_months():
    text = "\n".join(
        [
            "SGS Prices and Yields",
            ",,10-Year Bond Yield",
            "2019,Jan,2.15",
            ",Feb,2.10",
            "2020,Jan,1.61",
        ]
    )
    series = parse_sgs_10y(text)
    assert series.loc[pd.Timestamp("2019-02-01")] == 2.10
    assert series.loc[pd.Timestamp("2020-01-01")] == 1.61


def test_the_leaf_uses_the_previous_month():
    months = pd.date_range("2018-01-01", periods=40, freq="MS")
    yield_10y = pd.Series(range(40), index=months, dtype=float)
    credit = pd.Series(range(40, 80), index=months, dtype=float)
    leaves = assign_leaves(yield_10y, credit, window=36, min_periods=36)
    # Month t compares month t-1 with the median of months before t-1,
    # then the published leaf is shifted once more so month t uses month t-1's label.
    assert leaves.index.min() >= months[0] + pd.DateOffset(months=37)
    assert leaves["high_yield"].dtype == bool


def test_the_displayed_month_matches_the_leaf():
    months = pd.date_range("2018-01-01", periods=40, freq="MS")
    yield_10y = pd.Series(range(40), index=months, dtype=float)
    credit = pd.Series(range(40, 80), index=months, dtype=float)
    yields = pd.DataFrame({HY_TICKER: credit + 3.0, IG_TICKER: 3.0}, index=months)
    leaves = assign_leaves(yield_10y, credit_spread(yields))
    facts = month_facts(yield_10y, yields)
    assert facts["high_yield"].equals(leaves["high_yield"])
    assert facts["wide_credit"].equals(leaves["wide_credit"])
    assert facts["hy"].iloc[-1] - facts["ig"].iloc[-1] == facts["credit"].iloc[-1]
