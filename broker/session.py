"""Build the live sleeve, target weights, and unsent orders."""

from __future__ import annotations

import pandas as pd

from broker.plan import (
    build_rebalance_plan,
    latest_prices,
    load_target_weights,
    parked_value,
    sleeve_weights,
    tradable_orders,
    update_parked_flow,
    update_withdrawal,
)
from broker.quotes import download_tiger_closes
from broker.tiger_client import market_book, trade_client
from config import normalize_market
from paths import get_paths


def overlay_prices(market: str, weights: pd.Series, book: dict) -> pd.Series:
    prices = latest_prices(market)
    start = (pd.Timestamp.today().normalize() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    fresh = download_tiger_closes(list(weights.index), start, market)
    if fresh is not None and not fresh.empty:
        prices.update(fresh.ffill().iloc[-1].dropna())
    live = pd.Series(book["prices"], dtype=float)
    return live.combine_first(prices)


def prepare_rebalance(market: str = "us") -> dict:
    """Current account sleeve and the orders that would rebalance it. Nothing is sent."""
    market = normalize_market(market)
    weights = sleeve_weights(load_target_weights(market))
    book = market_book(trade_client(), market)
    prices = overlay_prices(market, weights, book)
    positions = book["positions"]
    parked = parked_value(positions, prices)
    _, processed_dir = get_paths(market)
    parked_flow = update_parked_flow(processed_dir / "parked_ledger.json", positions, prices)
    withdrawn, withdrawal = update_withdrawal(
        processed_dir / "cash_ledger.json", book["cash"], positions, prices
    )
    sleeve = float(book["asset_value"]) - parked - parked_flow
    plan = build_rebalance_plan(weights, prices, positions, sleeve, market)
    orders = tradable_orders(plan)
    plan.to_csv(processed_dir / "tiger_rebalance.csv", index=False)
    return {
        "market": market,
        "currency": book["currency"],
        "cash": float(book["cash"]),
        "stock_value": float(book["stock_value"]),
        "parked": parked,
        "parked_flow": parked_flow,
        "withdrawn": withdrawn,
        "withdrawal": withdrawal,
        "sleeve": sleeve,
        "plan": plan,
        "orders": orders,
    }
