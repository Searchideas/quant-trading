import numpy as np
import pandas as pd

from factors.momentum import MomentumFactor
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
