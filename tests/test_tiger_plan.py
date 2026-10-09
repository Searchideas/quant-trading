import pandas as pd
import pytest

from broker.tiger_client import place_orders
from broker.plan import (
    build_rebalance_plan,
    tiger_to_yahoo,
    tradable_orders,
    yahoo_to_tiger,
)


def test_place_orders_retries_a_rate_limit(monkeypatch):
    calls = {"n": 0}

    class Client:
        _account = "x"

        def place_order(self, order):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("code=1200 msg=too_many_requests:error.requestRateExceedLimit")
            return 99

    monkeypatch.setattr(
        "broker.tiger_client._import_tiger",
        lambda: {
            "stock_contract": lambda symbol, currency: (symbol, currency),
            "market_order": lambda *args: args,
        },
    )
    monkeypatch.setattr("broker.tiger_client.time.sleep", lambda seconds: None)
    orders = pd.DataFrame([{"tiger_symbol": "CRM", "action": "BUY", "delta_shares": 2}])
    sent = place_orders(Client(), orders, "USD")
    assert calls["n"] == 3
    assert sent == [{"tiger_symbol": "CRM", "action": "BUY", "quantity": 2, "order_id": 99}]


def test_us_class_share_symbols():
    assert yahoo_to_tiger("BRK-B", "us") == "BRK.B"
    assert tiger_to_yahoo("BRK.B", "us") == "BRK-B"
    assert yahoo_to_tiger("ADI", "us") == "ADI"


def test_sg_suffix_symbols():
    assert yahoo_to_tiger("D05.SI", "sg") == "D05"
    assert tiger_to_yahoo("D05", "sg") == "D05.SI"


def test_rebalance_sells_before_buys_and_skips_missing_prices():
    weights = pd.Series({"AAA": 0.5, "BBB": 0.5, "CCC": 0.0})
    prices = pd.Series({"AAA": 10.0, "BBB": 20.0})
    positions = {"BBB": 10, "CCC": 4}
    plan = build_rebalance_plan(weights, prices, positions, nav=1_000.0, market="us")
    orders = tradable_orders(plan)

    assert list(orders["ticker"]) == ["CCC", "AAA", "BBB"]
    assert list(orders["action"]) == ["SELL", "BUY", "BUY"]
    assert list(orders["delta_shares"]) == [-4, 50, 15]
    assert plan.loc[plan["ticker"] == "CCC", "action"].item() == "SELL"
    # CCC is an existing position leaving the book, so it is sold without a price.
    # A target name with no price is skipped instead.
    missing = build_rebalance_plan(
        pd.Series({"ZZZ": 1.0}),
        pd.Series(dtype=float),
        {},
        nav=1_000.0,
        market="us",
    )
    assert missing.loc[0, "action"] == "SKIP"
    assert tradable_orders(missing).empty


def test_excluded_names_stay_put():
    weights = pd.Series({"AAA": 1.0, "VOO": 0.0})
    prices = pd.Series({"AAA": 10.0, "VOO": 100.0, "TSM": 50.0})
    positions = {"VOO": 2, "TSM": 4}
    plan = build_rebalance_plan(
        weights, prices, positions, nav=1_000.0, market="us", exclude={"VOO", "TSM"}
    )
    held = plan[plan["ticker"].isin(["VOO", "TSM"])]
    assert list(held["action"]) == ["HOLD", "HOLD"]
    assert list(held["delta_shares"]) == [0, 0]
    assert list(tradable_orders(plan)["ticker"]) == ["AAA"]


def test_selling_a_parked_share_is_not_counted_twice(tmp_path):
    from broker.plan import update_parked_flow

    path = tmp_path / "parked_ledger.json"
    prices = pd.Series({"VOO": 712.03, "TSM": 400.0})
    assert update_parked_flow(path, {"VOO": 21, "TSM": 15}, prices) == 0.0
    flow = update_parked_flow(path, {"VOO": 20, "TSM": 15}, prices)
    assert flow == pytest.approx(712.03)
    assert update_parked_flow(path, {"VOO": 20, "TSM": 15}, prices) == pytest.approx(712.03)


def test_converting_usd_cash_is_a_withdrawal(tmp_path):
    from broker.plan import update_withdrawal

    path = tmp_path / "cash_ledger.json"
    prices = pd.Series({"CRM": 200.0})
    held = {"CRM": 17}
    assert update_withdrawal(path, 455.74, held, prices) == (0.0, 0.0)
    cumulative, taken = update_withdrawal(path, 0.0, held, prices)
    assert taken == pytest.approx(455.74)
    assert cumulative == pytest.approx(455.74)
    assert update_withdrawal(path, 0.0, held, prices) == pytest.approx((455.74, 0.0))


def test_buying_shares_is_not_a_withdrawal(tmp_path):
    from broker.plan import update_withdrawal

    path = tmp_path / "cash_ledger.json"
    prices = pd.Series({"CRM": 50.0})
    update_withdrawal(path, 1000.0, {}, prices)
    cumulative, taken = update_withdrawal(path, 500.0, {"CRM": 10}, prices)
    assert taken == 0.0
    assert cumulative == 0.0


def test_gld_is_parked_by_default():
    weights = pd.Series({"AAA": 1.0})
    prices = pd.Series({"AAA": 10.0, "GLD": 100.0})
    positions = {"GLD": 6}
    plan = build_rebalance_plan(weights, prices, positions, nav=1_000.0, market="us")
    gld = plan[plan["ticker"] == "GLD"].iloc[0]
    assert gld["action"] == "HOLD"
    assert int(gld["delta_shares"]) == 0
    assert list(tradable_orders(plan)["ticker"]) == ["AAA"]
