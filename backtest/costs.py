"""Tiger Brokers Singapore fees for US stocks and ETFs.

Each weight change is one order. Share count is the traded dollars divided by
that day's close. A fee is raised to its minimum, then cut at its maximum.
Orders under one share waive commission and pay a 1% platform fee capped at $1.
FINRA's trading-activity fee is charged on sells before 1 Oct 2026 and is zero
after that. Singapore GST of 9% is applied to the fee total.
"""

from __future__ import annotations

import pandas as pd

COMMISSION_PER_SHARE = 0.005
COMMISSION_MIN = 0.99
PLATFORM_PER_SHARE = 0.005
PLATFORM_MIN = 1.0
FEE_CAP_RATE = 0.005
FRACTIONAL_PLATFORM_RATE = 0.01
FRACTIONAL_PLATFORM_CAP = 1.0
SETTLEMENT_PER_SHARE = 0.003
SEC_SELL_RATE = 0.0000206
SEC_MIN = 0.01
TAF_PER_SHARE = 0.000195
TAF_MIN = 0.01
TAF_MAX = 9.79
TAF_ENDS = pd.Timestamp("2026-10-01")
CAT_PER_SHARE = 0.000003
GST_RATE = 0.09


def _bounded(amount: float, minimum: float | None, cap: float | None) -> float:
    if minimum is not None:
        amount = max(amount, minimum)
    if cap is not None:
        amount = min(amount, cap)
    return amount


def order_fee(shares: float, trade_value: float, side: str, trade_date: pd.Timestamp) -> float:
    """Dollar fee for one US stock order, including 9% GST."""
    shares = abs(float(shares))
    trade_value = abs(float(trade_value))
    if shares == 0 or trade_value == 0:
        return 0.0

    cap = FEE_CAP_RATE * trade_value
    if shares < 1:
        commission = 0.0
        platform = min(FRACTIONAL_PLATFORM_RATE * trade_value, FRACTIONAL_PLATFORM_CAP)
    else:
        commission = _bounded(COMMISSION_PER_SHARE * shares, COMMISSION_MIN, cap)
        platform = _bounded(PLATFORM_PER_SHARE * shares, PLATFORM_MIN, cap)
    settlement = min(SETTLEMENT_PER_SHARE * shares, cap)
    cat = CAT_PER_SHARE * shares
    sec = 0.0
    taf = 0.0
    if side == "sell":
        sec = max(SEC_SELL_RATE * trade_value, SEC_MIN)
        if pd.Timestamp(trade_date).tz_localize(None).normalize() < TAF_ENDS:
            taf = _bounded(TAF_PER_SHARE * shares, TAF_MIN, TAF_MAX)
    return (commission + platform + settlement + cat + sec + taf) * (1 + GST_RATE)


def calculate_costs(
    weightage: pd.DataFrame,
    prev_equity: pd.Series,
    prices: pd.DataFrame,
) -> pd.Series:
    """Tiger fee for each name whose weight changes, sized off prior equity."""
    if weightage.empty:
        return pd.Series(dtype=float)

    delta = weightage - weightage.shift(1).fillna(0.0)
    prices = prices.reindex(index=weightage.index, columns=weightage.columns)
    equity = prev_equity.reindex(weightage.index)
    costs = pd.Series(0.0, index=weightage.index)

    for date in weightage.index:
        capital = equity.loc[date]
        if pd.isna(capital) or float(capital) <= 0:
            continue
        day = delta.loc[date]
        traded = day[day != 0]
        total = 0.0
        for ticker, change in traded.items():
            price = prices.at[date, ticker]
            if pd.isna(price) or float(price) <= 0:
                continue
            trade_value = abs(float(change)) * float(capital)
            shares = trade_value / float(price)
            side = "buy" if float(change) > 0 else "sell"
            total += order_fee(shares, trade_value, side, date)
        costs.loc[date] = total
    return costs
