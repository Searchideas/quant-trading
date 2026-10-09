"""Size the latest strategy book to the Tiger account and print the orders.

Dry run (no orders):

    python -m broker --market us

Send market orders to the account in tiger_openapi_config.properties:

    python -m broker --market us --place
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from broker.plan import EXCLUDED_TICKERS, is_excluded
from broker.session import prepare_rebalance
from broker.tiger_client import _market_currency, place_orders, trade_client
from config import normalize_market
from paths import get_paths


def run(market: str, place: bool) -> None:
    market = normalize_market(market)
    ready = prepare_rebalance(market)
    plan = ready["plan"]
    orders = ready["orders"]
    _, processed_dir = get_paths(market)

    currency = ready["currency"]
    print(f"[{market}] {currency} cash: {ready['cash']:,.2f}")
    print(f"[{market}] {currency} stocks: {ready['stock_value']:,.2f}")
    print(
        f"[{market}] Left unchanged ({', '.join(sorted(EXCLUDED_TICKERS))}): "
        f"{ready['parked']:,.2f} {currency}"
    )
    print(f"[{market}] Asset value used for targets: {ready['sleeve']:,.2f} {currency}")
    print(f"Saved plan to {processed_dir / 'tiger_rebalance.csv'}")

    targets = plan[plan["weight"] > 0][
        ["ticker", "weight", "price", "target_shares", "current_shares", "target_value"]
    ]
    print("\nTarget shares")
    print(targets.to_string(index=False))
    untouched = plan[plan["ticker"].map(is_excluded)]
    if not untouched.empty:
        print("\nLeft unchanged")
        print(
            untouched[["ticker", "current_shares", "price", "target_value"]].to_string(index=False)
        )
    if orders.empty:
        print("\nNo trades. Holdings already match the target book.")
        return

    print("\nOrders to send")
    print(
        orders[
            ["action", "tiger_symbol", "delta_shares", "current_shares", "target_shares", "price", "order_value"]
        ].to_string(index=False)
    )
    if not place:
        print("\nDry run only. Nothing was sent. Re-run with --place to send these market orders.")
        return

    _, _, currency = _market_currency(market)
    print(f"\nSending {len(orders)} market orders ({currency}).")
    place_orders(trade_client(), orders, currency)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebalance a Tiger account to strategy weights")
    parser.add_argument("--market", default="us", choices=["us", "sg"])
    parser.add_argument(
        "--place",
        action="store_true",
        help="Send market orders. Without this flag the command only prints the plan.",
    )
    args = parser.parse_args()
    run(market=args.market, place=args.place)


if __name__ == "__main__":
    main()
