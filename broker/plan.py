"""Turn strategy weights into a share-level rebalance plan."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from paths import get_paths

_CLASS_SHARE = re.compile(r"^[A-Z]+[.-][A-Z]$")

# Left untouched. Their market value is not used to size the strategy book.
EXCLUDED_TICKERS = frozenset({"VOO", "TSM", "GLD", "D05", "D05.SI", "O39", "O39.SI"})


def yahoo_to_tiger(symbol: str, market: str) -> str:
    """Map a Yahoo symbol to the symbol Tiger expects."""
    symbol = symbol.strip()
    if market == "sg" and symbol.endswith(".SI"):
        return symbol[: -len(".SI")]
    if market == "us" and _CLASS_SHARE.match(symbol) and "-" in symbol:
        return symbol.replace("-", ".")
    return symbol


def tiger_to_yahoo(symbol: str, market: str) -> str:
    """Map a Tiger symbol back to the Yahoo symbol used in this project."""
    symbol = symbol.strip()
    if market == "sg" and not symbol.endswith(".SI"):
        return f"{symbol}.SI"
    if market == "us" and _CLASS_SHARE.match(symbol) and "." in symbol:
        return symbol.replace(".", "-")
    return symbol


def load_target_weights(market: str) -> pd.Series:
    """Latest target weights written by the strategy."""
    _, processed_dir = get_paths(market)
    path = processed_dir / "weights_now.csv"
    frame = pd.read_csv(path)
    if "ticker" in frame.columns:
        frame = frame.set_index("ticker")
    weights = frame["weight"].astype(float)
    weights = weights[weights > 1e-12]
    weights.index = weights.index.astype(str)
    return weights


def latest_prices(market: str) -> pd.Series:
    """Last available close for each Yahoo ticker."""
    raw_dir, _ = get_paths(market)
    prices = pd.read_csv(raw_dir / "price_data.csv", index_col=0, parse_dates=True)
    last = prices.ffill().iloc[-1].dropna()
    last.index = last.index.astype(str)
    return last


def is_excluded(ticker: str, exclude: frozenset[str] | set[str] | None = None) -> bool:
    blocked = EXCLUDED_TICKERS if exclude is None else exclude
    return str(ticker).upper() in {name.upper() for name in blocked}


def parked_value(
    positions: dict[str, float],
    prices: pd.Series,
    exclude: frozenset[str] | set[str] | None = None,
) -> float:
    """Market value of names that stay outside the strategy sleeve."""
    total = 0.0
    for ticker, shares in positions.items():
        if not is_excluded(ticker, exclude):
            continue
        price = prices.get(ticker)
        if price is None or pd.isna(price) or float(price) <= 0:
            continue
        total += abs(float(shares)) * float(price)
    return total


def excluded_shares(
    positions: dict[str, float],
    exclude: frozenset[str] | set[str] | None = None,
) -> dict[str, float]:
    """Share counts for names kept outside the strategy sleeve."""
    return {
        str(ticker): float(shares)
        for ticker, shares in positions.items()
        if is_excluded(ticker, exclude)
    }


def update_parked_flow(
    path: Path,
    positions: dict[str, float],
    prices: pd.Series,
    exclude: frozenset[str] | set[str] | None = None,
) -> float:
    """Cash that moved from parked names into the sleeve since the ledger started.

    Selling one parked share adds the proceeds. Buying one removes them.
    The first call stores the current share counts and returns 0.
    """
    current = excluded_shares(positions, exclude)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        previous = {str(ticker): float(shares) for ticker, shares in (saved.get("shares") or {}).items()}
        flow = float(saved.get("flow") or 0.0)
    else:
        previous = {}
        flow = 0.0
        saved = None
    if saved is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"shares": current, "flow": 0.0}), encoding="utf-8")
        return 0.0

    updated = dict(previous)
    for ticker in set(previous) | set(current):
        before = float(previous.get(ticker, 0.0))
        after = float(current.get(ticker, 0.0))
        delta = after - before
        if abs(delta) < 1e-9:
            updated[ticker] = after
            continue
        price = prices.get(ticker)
        if price is None or pd.isna(price) or float(price) <= 0:
            continue
        flow += -delta * float(price)
        updated[ticker] = after
    path.write_text(json.dumps({"shares": updated, "flow": flow}), encoding="utf-8")
    return flow


def strategy_shares(
    positions: dict[str, float],
    exclude: frozenset[str] | set[str] | None = None,
) -> dict[str, float]:
    """Share counts that belong to the strategy sleeve."""
    return {
        str(ticker): float(shares)
        for ticker, shares in positions.items()
        if not is_excluded(ticker, exclude) and abs(float(shares)) >= 1e-9
    }


def update_withdrawal(
    path: Path,
    usd_cash: float,
    positions: dict[str, float],
    prices: pd.Series,
    exclude: frozenset[str] | set[str] | None = None,
) -> tuple[float, float]:
    """Cumulative cash taken out of the sleeve, and the amount taken this call.

    Share trades explain their own cash. Cash that leaves with no matching
    trade, including a conversion into another currency, is a withdrawal.
    A later call with the same balances does not add it again.
    """
    current = strategy_shares(positions, exclude)
    usd_cash = float(usd_cash)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"usd_cash": usd_cash, "shares": current, "withdrawn": 0.0}),
            encoding="utf-8",
        )
        return 0.0, 0.0

    saved = json.loads(path.read_text(encoding="utf-8"))
    previous = {str(ticker): float(shares) for ticker, shares in (saved.get("shares") or {}).items()}
    prev_cash = float(saved.get("usd_cash") or 0.0)
    withdrawn = float(saved.get("withdrawn") or 0.0)
    updated = dict(previous)
    trade_cash = 0.0
    for ticker in set(previous) | set(current):
        before = float(previous.get(ticker, 0.0))
        after = float(current.get(ticker, 0.0))
        delta = after - before
        if abs(delta) < 1e-9:
            updated[ticker] = after
            continue
        price = prices.get(ticker)
        if price is None or pd.isna(price) or float(price) <= 0:
            return withdrawn, 0.0
        trade_cash += -delta * float(price)
        updated[ticker] = after
    residual = usd_cash - (prev_cash + trade_cash)
    taken = -residual if residual < -0.5 else 0.0
    withdrawn += taken
    path.write_text(
        json.dumps({"usd_cash": usd_cash, "shares": updated, "withdrawn": withdrawn}),
        encoding="utf-8",
    )
    return withdrawn, taken


def sleeve_weights(
    weights: pd.Series,
    exclude: frozenset[str] | set[str] | None = None,
) -> pd.Series:
    """Drop excluded names and rescale the remaining weights to 1."""
    kept = weights[~weights.index.map(lambda ticker: is_excluded(str(ticker), exclude))]
    if kept.empty or float(kept.sum()) <= 0:
        return kept
    return kept / float(kept.sum())


def build_rebalance_plan(
    weights: pd.Series,
    prices: pd.Series,
    positions: dict[str, float],
    nav: float,
    market: str,
    exclude: frozenset[str] | set[str] | None = None,
) -> pd.DataFrame:
    """Share differences between the target book and current positions.

    ``positions`` is keyed by Yahoo ticker. Sells come before buys so cash
    is freed before new names are purchased.
    """
    if nav <= 0:
        raise ValueError(f"Account value must be positive, got {nav}")

    names = set(weights.index) | set(positions)
    rows: list[dict] = []
    for ticker in sorted(names):
        weight = float(weights.get(ticker, 0.0))
        current = int(round(float(positions.get(ticker, 0.0))))
        price = prices.get(ticker)
        price_ok = price is not None and pd.notna(price) and float(price) > 0
        price_value = float(price) if price_ok else float("nan")
        if is_excluded(ticker, exclude):
            target = current
            delta = 0
            action = "HOLD"
            weight = 0.0
        elif not price_ok and current > 0 and weight <= 1e-12:
            target = 0
            delta = -current
            action = "SELL"
        elif not price_ok and weight > 1e-12:
            target = 0
            delta = 0
            action = "SKIP"
        else:
            target = int(round(weight * nav / float(price))) if price_ok else current
            delta = target - current
            if delta > 0:
                action = "BUY"
            elif delta < 0:
                action = "SELL"
            else:
                action = "HOLD"
        rows.append(
            {
                "ticker": ticker,
                "tiger_symbol": yahoo_to_tiger(ticker, market),
                "weight": weight,
                "price": price_value,
                "target_shares": target,
                "current_shares": current,
                "delta_shares": delta,
                "target_value": target * price_value if price_ok else float("nan"),
                "order_value": delta * price_value if price_ok else float("nan"),
                "action": action,
            }
        )

    plan = pd.DataFrame(rows)
    if plan.empty:
        return plan
    order = {"SELL": 0, "BUY": 1, "SKIP": 2, "HOLD": 3}
    plan["_ord"] = plan["action"].map(order)
    plan = plan.sort_values(["_ord", "ticker"]).drop(columns="_ord").reset_index(drop=True)
    return plan


def tradable_orders(plan: pd.DataFrame) -> pd.DataFrame:
    """Rows that would be sent to the broker."""
    if plan.empty:
        return plan
    return plan[plan["action"].isin(["BUY", "SELL"]) & (plan["delta_shares"] != 0)].reset_index(
        drop=True
    )
