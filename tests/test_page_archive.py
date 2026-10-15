"""The saved book keeps old sessions, and the mix search stops at 2022."""

import pandas as pd

from web.page_archive import extend_records, in_training


def test_training_window_is_2017_through_2022():
    assert in_training("2017-01-03")
    assert in_training("2022-12-30")
    assert in_training("2022-12-31")
    assert not in_training("2016-12-30")
    assert not in_training("2023-01-03")


def test_append_does_not_change_a_saved_day():
    days = pd.to_datetime(["2023-01-03", "2023-01-04"])
    history = pd.DataFrame(
        {
            "book": [1000.0, 1010.0],
            "spy": [1000.0, 1005.0],
            "spy_close": [100.0, 100.5],
            "cost": [0.0, 0.0],
            "turnover": [0.0, 0.0],
            "leaf": ["1,0", "1,0"],
            "holdings": ["AAA 100.0", "AAA 100.0"],
        },
        index=days,
    )
    weights = pd.DataFrame({"AAA": [1.0, 1.0]}, index=days)
    prices = pd.DataFrame({"AAA": [100.0, 100.0]}, index=days)
    new_day = pd.Timestamp("2023-01-05")
    new_weights = pd.DataFrame({"AAA": [1.0]}, index=[new_day])
    new_prices = pd.DataFrame({"AAA": [110.0]}, index=[new_day])
    new_spy = pd.Series([101.5], index=[new_day])
    new_leaves = pd.Series(["0,1"], index=[new_day])

    updated, updated_weights, _prices = extend_records(
        history, weights, prices, new_weights, new_prices, new_spy, new_leaves
    )

    assert updated.loc[days[0], "book"] == 1000.0
    assert updated.loc[days[1], "book"] == 1010.0
    assert updated.loc[days[0], "leaf"] == "1,0"
    assert updated_weights.loc[days[0], "AAA"] == 1.0
    assert abs(updated.loc[new_day, "book"] - 1010.0 * 1.10) < 1e-6
    assert updated.loc[new_day, "leaf"] == "0,1"
    assert abs(updated.loc[new_day, "spy"] - 1005.0 * (101.5 / 100.5)) < 1e-6
