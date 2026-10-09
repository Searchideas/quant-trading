import pandas as pd
import pytest

from analytics.performance import month_lead


def test_month_lead_needs_more_than_half_the_months():
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01"])
    strategy = pd.Series([0.10, 0.0, -0.05], index=index)
    benchmark = pd.Series([0.0, 0.0, 0.10], index=index)
    share, median = month_lead(strategy, benchmark)
    assert share == 0.5
    assert median == pytest.approx((0.10 - 0.15) / 2)
