import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine
from factors.dividend import DividendFactor
from factors.momentum import MomentumFactor
from factors.value import ValueFactor
from factors.volatility import VolatilityFactor
from strategy.factor_strategy import hide_unlisted, listed_span


def test_a_name_has_no_factor_before_its_first_price():
    prices = pd.DataFrame(
        {"NEW": [np.nan] * 40 + [10, 11, 12, 13, 14, 15, 16], "OLD": range(47)},
        index=pd.bdate_range("2020-01-01", periods=47),
        dtype=float,
    )
    momentum = MomentumFactor(prices).calculate_momentum(lookback=5)
    volatility = VolatilityFactor(prices).calculate_volatility(lookback=5)
    assert momentum["NEW"].iloc[:40].isna().all()
    assert volatility["NEW"].iloc[:40].isna().all()
    assert volatility["NEW"].iloc[-1] > 0


def test_unlisted_score_cannot_be_selected():
    index = pd.bdate_range("2022-01-03", periods=4)
    prices = pd.DataFrame({"SPCX": [np.nan, np.nan, 10.0, 11.0]}, index=index)
    score = pd.DataFrame({"SPCX": [1.0, 1.0, 1.0, 1.0]}, index=index)
    hidden = hide_unlisted(score, prices)
    assert hidden["SPCX"].iloc[:2].isna().all()
    assert listed_span(prices)["SPCX"].tolist() == [False, False, True, True]


def test_book_value_is_unused_until_forty_days_after_quarter_end():
    book = pd.DataFrame(
        {"AAA": [10.0, 20.0]},
        index=pd.to_datetime(["2020-03-31", "2020-06-30"]),
    )
    prices = pd.DataFrame(
        {"AAA": 100.0},
        index=pd.bdate_range("2020-03-31", "2020-08-14"),
    )
    value = ValueFactor(prices, book).calculate_value_factor()
    assert pd.isna(value.loc["2020-05-08", "AAA"])
    assert value.loc["2020-05-11", "AAA"] == pytest.approx(0.1)
    assert value.loc["2020-08-07", "AAA"] == pytest.approx(0.1)
    assert value.loc["2020-08-10", "AAA"] == pytest.approx(0.2)


def test_another_names_blank_row_does_not_erase_the_last_figure():
    book = pd.DataFrame(
        {"AAA": [10.0, np.nan, 20.0], "BBB": [np.nan, 5.0, np.nan]},
        index=pd.to_datetime(["2020-03-31", "2020-04-30", "2020-06-30"]),
    )
    prices = pd.DataFrame(
        {"AAA": 100.0, "BBB": 50.0},
        index=pd.bdate_range("2020-03-31", "2020-08-14"),
    )
    value = ValueFactor(prices, book).calculate_value_factor()
    assert pd.isna(value.loc["2020-05-08", "AAA"])
    assert value.loc["2020-06-09", "AAA"] == pytest.approx(0.1)
    assert value.loc["2020-08-07", "AAA"] == pytest.approx(0.1)
    assert value.loc["2020-08-10", "AAA"] == pytest.approx(0.2)
    assert pd.isna(value.loc["2020-06-08", "BBB"])
    assert value.loc["2020-06-09", "BBB"] == pytest.approx(0.1)

    dividends = pd.DataFrame(
        {"AAA": [1.0, np.nan, 2.0], "BBB": [np.nan, 0.4, np.nan]},
        index=pd.to_datetime(["2020-05-01", "2020-05-15", "2020-07-01"]),
    )
    dividend_prices = pd.DataFrame(
        {"AAA": 100.0, "BBB": 50.0},
        index=pd.bdate_range("2020-04-30", "2020-07-15"),
    )
    dividend = DividendFactor(dividend_prices, dividends).calculate_dividend_factor()
    assert pd.isna(dividend.loc["2020-04-30", "AAA"])
    assert dividend.loc["2020-05-01", "AAA"] == pytest.approx(0.01)
    assert dividend.loc["2020-06-15", "AAA"] == pytest.approx(0.01)
    assert dividend.loc["2020-07-15", "AAA"] == pytest.approx(0.02)
    assert pd.isna(dividend.loc["2020-05-14", "BBB"])
    assert dividend.loc["2020-06-15", "BBB"] == pytest.approx(0.008)


def test_the_signal_close_does_not_earn_the_next_move():
    dates = pd.to_datetime(["2026-10-05", "2026-10-06", "2026-10-07"])
    prices = pd.DataFrame({"AAA": [100.0, 110.0, 121.0]}, index=dates)
    weights = pd.DataFrame({"AAA": [1.0, 1.0, 1.0]}, index=dates)
    result = BacktestEngine(prices, weights, initial_capital=100).run()
    assert result.loc[dates[0], "gross_return"] == 0
    assert result.loc[dates[1], "gross_return"] == 0
    assert result.loc[dates[2], "gross_return"] == pytest.approx(0.10)
