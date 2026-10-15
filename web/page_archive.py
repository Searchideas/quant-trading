"""Saved book path. A later run appends new sessions and leaves old rows in place."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backtest.costs import calculate_costs
from strategy.factor_strategy import select_sector_best_weights

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "us" / "processed" / "page_store"
TRAIN_START = pd.Timestamp("2017-01-01")
TRAIN_END = pd.Timestamp("2023-01-01")


def naive_index(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    return idx.normalize()


def in_training(day) -> bool:
    """True on the sessions used to choose the four mixes."""
    stamp = pd.Timestamp(day)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(None)
    stamp = stamp.normalize()
    return TRAIN_START <= stamp < TRAIN_END


def ready() -> bool:
    return (STORE / "history.csv").exists() and (STORE / "mixes.json").exists()


def load_mixes() -> dict[tuple[bool, bool], tuple[float, float, float, float]]:
    payload = json.loads((STORE / "mixes.json").read_text(encoding="utf-8"))
    mixes = {}
    for key, values in payload["mixes"].items():
        high, wide = key.split(",")
        mixes[(high == "1", wide == "1")] = tuple(float(item) for item in values)
    return mixes


def load_history() -> pd.DataFrame:
    history = pd.read_csv(STORE / "history.csv", index_col=0, parse_dates=True)
    history.index = naive_index(history.index)
    return history.sort_index()


def load_weights() -> pd.DataFrame:
    weights = pd.read_csv(STORE / "weights.csv", index_col=0, parse_dates=True).fillna(0.0)
    weights.index = naive_index(weights.index)
    return weights.sort_index()


def load_prices() -> pd.DataFrame:
    prices = pd.read_csv(STORE / "prices.csv", index_col=0, parse_dates=True)
    prices.index = naive_index(prices.index)
    return prices.sort_index()


def load_categories() -> pd.DataFrame:
    return pd.read_csv(STORE / "categories.csv", index_col=0)


def sector_caps(categories: pd.DataFrame) -> pd.Series:
    return categories.groupby("sector")["market_cap"].sum().sort_values(ascending=False)


def save_run(
    history: pd.DataFrame,
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    categories: pd.DataFrame,
    mixes: dict[tuple[bool, bool], tuple[float, float, float, float]],
) -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    history.to_csv(STORE / "history.csv")
    weights.to_csv(STORE / "weights.csv")
    prices.to_csv(STORE / "prices.csv")
    categories.to_csv(STORE / "categories.csv")
    payload = {
        "train_start": TRAIN_START.strftime("%Y-%m-%d"),
        "train_end": TRAIN_END.strftime("%Y-%m-%d"),
        "mixes": {
            f"{int(high)},{int(wide)}": list(mix)
            for (high, wide), mix in mixes.items()
        },
    }
    (STORE / "mixes.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def leaf_key(text: str) -> tuple[bool, bool]:
    high, wide = text.split(",")
    return high == "1", wide == "1"


def leaf_text(key: tuple[bool, bool] | None) -> str:
    if key is None:
        return ""
    return f"{int(key[0])},{int(key[1])}"


def held_names(row: pd.Series) -> tuple[str, ...]:
    kept = row[row > 0.001]
    return tuple(sorted(kept.index.astype(str)))


def day_weights(
    score: pd.Series,
    categories: pd.DataFrame,
    caps: pd.Series,
) -> pd.Series:
    """Target book for one close. Same rule as the full weight history on a single row."""
    frame = pd.DataFrame([score])
    weights = select_sector_best_weights(
        frame,
        categories,
        caps,
        max_sector_weight=1.0,
        top_n_per_sector=2,
    )
    return weights.iloc[0]


def holdings_text(row: pd.Series) -> str:
    held = row[row > 0.001].sort_values(ascending=False)
    return "|".join(f"{ticker} {weight * 100:.1f}" for ticker, weight in held.items())


def _fee(weights: pd.DataFrame, prices: pd.DataFrame, day: pd.Timestamp, capital: float) -> float:
    """Fee for the fill that lands on this close, sized from the prior equity."""
    filled = weights.shift(1)
    window = filled.index[filled.index <= day]
    if len(window) < 2:
        return 0.0
    frame = filled.loc[window[-2:]]
    equity = pd.Series(capital, index=frame.index)
    costs = calculate_costs(frame, equity, prices)
    value = float(costs.iloc[-1])
    return value if pd.notna(value) else 0.0


def extend_records(
    history: pd.DataFrame,
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    new_weights: pd.DataFrame,
    new_prices: pd.DataFrame,
    new_spy: pd.Series,
    new_leaves: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Append sessions after the saved history. Existing rows stay as stored."""
    last = history.index[-1]
    fresh = new_prices.reindex(columns=list(prices.columns))
    fresh.index = naive_index(fresh.index)
    new_weights = new_weights.copy()
    new_weights.index = naive_index(new_weights.index)
    added = fresh.index[fresh.index > last]
    if len(added) == 0:
        return history, weights, prices

    prices = prices.copy()
    weights = weights.copy()
    history = history.copy()
    for day in added:
        prices.loc[day] = fresh.loc[day]
        proposed = new_weights.reindex(index=[day], columns=weights.columns).fillna(0.0).iloc[0]
        previous = weights.iloc[-1]
        names = held_names(proposed)
        if not names or names == held_names(previous):
            weights.loc[day] = previous.to_numpy()
        else:
            weights.loc[day] = proposed.to_numpy()

    book_equity = float(history["book"].iloc[-1])
    spy_equity = float(history["spy"].iloc[-1])
    spy_close = float(history["spy_close"].iloc[-1])
    returns = prices.sort_index().pct_change(fill_method=None)
    spy = new_spy.copy()
    spy.index = naive_index(spy.index)
    for day in added:
        held = weights.shift(2).loc[day].fillna(0.0)
        move = returns.loc[day].fillna(0.0) if day in returns.index else 0.0
        gross = float((held * move).sum())
        if not pd.notna(gross):
            gross = 0.0
        cost = _fee(weights, prices, day, book_equity)
        net = gross - (cost / book_equity if book_equity else 0.0)
        book_equity *= 1.0 + net
        close = spy.loc[day] if day in spy.index else spy_close
        close = float(close) if pd.notna(close) else spy_close
        spy_move = (close / spy_close - 1.0) if spy_close else 0.0
        spy_equity *= 1.0 + spy_move
        spy_close = close
        changed = weights.loc[day] - weights.shift(1).loc[day]
        leaf = new_leaves.loc[day] if day in new_leaves.index else ""
        if not leaf:
            leaf = history["leaf"].iloc[-1]
        history.loc[day] = {
            "book": book_equity,
            "spy": spy_equity,
            "spy_close": spy_close,
            "cost": cost,
            "turnover": float(changed.abs().sum()) / 2.0,
            "leaf": leaf,
            "holdings": holdings_text(weights.loc[day]),
        }
    return history, weights, prices
