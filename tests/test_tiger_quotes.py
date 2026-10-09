import pandas as pd

from broker.quotes import bars_to_closes, close_mismatches, combine_closes


def test_bars_map_class_shares_to_yahoo_dates():
    bars = pd.DataFrame(
        {
            "symbol": ["BRK.B", "BRK.B"],
            "time": [1_700_000_000_000, 1_700_086_400_000],
            "close": [500.0, 505.0],
        }
    )
    closes = bars_to_closes(bars, "us")
    assert list(closes.columns) == ["BRK-B"]
    assert closes["BRK-B"].iloc[-1] == 505.0


def test_yahoo_fills_only_symbols_tiger_missed():
    tiger = pd.DataFrame({"ADI": [100.0, 101.0]}, index=pd.to_datetime(["2026-09-24", "2026-09-25"]))
    yahoo = pd.DataFrame(
        {"ADI": [90.0, 91.0], "IBM": [200.0, 201.0]},
        index=pd.to_datetime(["2026-09-24", "2026-09-25"]),
    )
    combined, filled = combine_closes(tiger, yahoo)
    assert filled == ["IBM"]
    assert combined.loc["2026-09-25", "ADI"] == 101.0
    assert combined.loc["2026-09-25", "IBM"] == 201.0


def test_dividend_rows_use_ex_date_and_yahoo_symbol():
    from broker.quotes import dividends_to_frame

    rows = pd.DataFrame(
        {
            "symbol": ["BRK.B", "AAPL"],
            "execute_date": ["2026-05-08", "2026-05-08"],
            "amount": [0.0, 0.26],
        }
    )
    wide = dividends_to_frame(rows, "us")
    assert wide.loc["2026-05-08", "AAPL"] == 0.26
    assert "BRK-B" in wide.columns


def test_mismatch_flags_a_one_percent_gap():
    index = pd.to_datetime(["2026-09-25"])
    tiger = pd.DataFrame({"ADI": [110.0], "T": [25.0]}, index=index)
    yahoo = pd.DataFrame({"ADI": [100.0], "T": [25.1]}, index=index)
    flagged = close_mismatches(tiger, yahoo, threshold=0.01)
    assert list(flagged["ticker"]) == ["ADI"]
