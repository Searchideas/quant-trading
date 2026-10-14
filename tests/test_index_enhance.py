import numpy as np
import pandas as pd
import pytest

from strategy.index_enhance import (
    fit_active,
    index_weights,
    review_sessions,
    selective_active,
    share_proxy,
    third_friday,
    tracking_error,
)


def test_third_friday_of_march_2024_is_the_15th():
    assert third_friday(2024, 3) == pd.Timestamp("2024-03-15")
    assert third_friday(2024, 9) == pd.Timestamp("2024-09-20")


def test_review_uses_the_session_on_or_before_the_third_friday():
    sessions = pd.to_datetime(["2024-03-14", "2024-03-15", "2024-03-18", "2024-09-19", "2024-09-20"])
    reviews = review_sessions(pd.DatetimeIndex(sessions))
    assert list(reviews) == [pd.Timestamp("2024-03-15"), pd.Timestamp("2024-09-20")]


def test_index_weight_uses_that_sessions_price_and_the_share_count():
    prices = pd.Series({"DBS": 10.0, "OCBC": 20.0})
    shares = pd.Series({"DBS": 5.0, "OCBC": 5.0})
    weights = index_weights(prices, shares)
    assert weights["DBS"] == pytest.approx(1 / 3)
    assert weights["OCBC"] == pytest.approx(2 / 3)


def test_share_proxy_is_current_market_cap_over_the_latest_price():
    shares = share_proxy(pd.Series({"DBS": 40.0}), pd.Series({"DBS": 200.0}))
    assert shares["DBS"] == pytest.approx(5.0)


def test_only_an_above_average_name_is_overweighted():
    index_w = pd.Series({"DBS": 0.5, "OCBC": 0.5})
    score = pd.Series({"DBS": 1.0, "OCBC": -1.0})
    active = selective_active(index_w, score, scale=0.1, max_active=1.0)
    assert active["DBS"] == pytest.approx(0.05)
    assert active["OCBC"] == pytest.approx(-0.05)
    assert active.sum() == pytest.approx(0.0)


def test_tilt_is_scaled_inside_the_tracking_error_budget():
    index_w = pd.Series({"A": 0.5, "B": 0.5})
    score = pd.Series({"A": 1.0, "B": -1.0})
    cov = pd.DataFrame(
        [[0.0004, 0.0], [0.0, 0.0004]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    active = fit_active(index_w, score, cov, budget=0.0125, max_active=1.0)
    assert tracking_error(active, cov) == pytest.approx(0.0125, rel=1e-3)
    assert active["A"] > 0
    assert active["B"] < 0
    assert active.sum() == pytest.approx(0.0)


def test_a_missing_price_drops_out_of_the_index_weight():
    prices = pd.Series({"DBS": 10.0, "NEW": np.nan})
    shares = pd.Series({"DBS": 5.0, "NEW": 5.0})
    weights = index_weights(prices, shares)
    assert weights["DBS"] == pytest.approx(1.0)
    assert weights["NEW"] == pytest.approx(0.0)
