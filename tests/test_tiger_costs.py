import pandas as pd
import pytest

from backtest.costs import GST_RATE, order_fee
from backtest.engine import BacktestEngine


def _with_gst(*parts: float) -> float:
    return sum(parts) * (1 + GST_RATE)


def test_full_share_buy_uses_minimums_when_they_are_under_the_cap():
    # 1 share at $400. The 0.5% cap is $2, so the $0.99 and $1 minimums apply.
    fee = order_fee(1, 400, "buy", pd.Timestamp("2026-10-06"))
    assert fee == pytest.approx(_with_gst(0.99, 1.0, 0.003, 0.000003))


def test_cap_replaces_the_minimum_on_a_small_order():
    # 1 share at $100. The 0.5% cap is $0.50, below both minimums.
    fee = order_fee(1, 100, "buy", pd.Timestamp("2026-10-06"))
    assert fee == pytest.approx(_with_gst(0.50, 0.50, 0.003, 0.000003))


def test_fractional_order_waives_commission():
    # 0.5 share, $50 traded. Platform fee is 1% capped at $1.
    fee = order_fee(0.5, 50, "buy", pd.Timestamp("2026-10-06"))
    assert fee == pytest.approx(_with_gst(0.50, 0.0015, 0.0000015))


def test_sell_before_october_2026_includes_sec_and_taf():
    fee = order_fee(10, 1000, "sell", pd.Timestamp("2026-09-30"))
    assert fee == pytest.approx(_with_gst(0.99, 1.0, 0.03, 0.00003, 0.0206, 0.01))


def test_sell_from_october_2026_drops_the_trading_activity_fee():
    fee = order_fee(10, 1000, "sell", pd.Timestamp("2026-10-01"))
    assert fee == pytest.approx(_with_gst(0.99, 1.0, 0.03, 0.00003, 0.0206))


def test_opening_book_pays_the_buy_fee():
    dates = pd.to_datetime(["2026-10-06", "2026-10-07"])
    prices = pd.DataFrame({"AAA": [100.0, 100.0]}, index=dates)
    weights = pd.DataFrame({"AAA": [1.0, 1.0]}, index=dates)
    result = BacktestEngine(prices, weights, initial_capital=400).run()
    opening_fee = order_fee(4, 400, "buy", dates[0])
    assert result.loc[dates[0], "cost_sgd"] == pytest.approx(opening_fee)
    assert result.loc[dates[0], "return"] == pytest.approx(-opening_fee / 400)
    assert result.loc[dates[1], "cost_sgd"] == 0
