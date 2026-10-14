"""Choose a factor mix for each Singapore yield leaf.

Momentum, value, low volatility, and dividend are tried in 10-point steps
that sum to 100. Each leaf keeps the mix with the highest information ratio
of the capped STI book on that leaf's own days. Days before the median
exists get their own mix, chosen the same way. The same days are then used
in the reported path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.index_enhance import (
    MAX_ACTIVE_WEIGHT,
    TRACKING_ERROR_BUDGET,
    daily_covariance,
    index_weights,
    review_sessions,
)

KEYS = ((False, False), (False, True), (True, False), (True, True), None)
MIN_DAYS = 20


def grid(step: int = 10) -> list[tuple[float, float, float, float]]:
    """Mixes of momentum, value, low volatility, and dividend that sum to 1."""
    mixes = []
    for momentum in range(0, 101, step):
        for value in range(0, 101 - momentum, step):
            for volatility in range(0, 101 - momentum - value, step):
                dividend = 100 - momentum - value - volatility
                mixes.append((momentum / 100, value / 100, volatility / 100, dividend / 100))
    return mixes


def information_ratio(excess: np.ndarray) -> float:
    """Annualized mean excess divided by its standard deviation."""
    path = excess[np.isfinite(excess)]
    if path.size < MIN_DAYS:
        return float("nan")
    deviation = float(path.std(ddof=0))
    if deviation == 0:
        return float("nan")
    return float(path.mean() / deviation * np.sqrt(252))


def _factor_panel(normalized: pd.DataFrame, members: list[str], sessions: pd.DatetimeIndex) -> np.ndarray:
    """Shape (days, names, 4) in momentum, value, low volatility, dividend order."""
    factors = ("momentum", "value", "volatility", "dividend")
    panel = np.zeros((len(sessions), len(members), len(factors)), dtype=float)
    aligned = normalized.reindex(sessions)
    for name_index, name in enumerate(members):
        for factor_index, factor in enumerate(factors):
            column = f"{name}_{factor}"
            if column in aligned.columns:
                panel[:, name_index, factor_index] = aligned[column].fillna(0.0).to_numpy(dtype=float)
    return panel


def _active_batch(
    index_w: np.ndarray,
    scores: np.ndarray,
    covariance: np.ndarray,
    budget: float = TRACKING_ERROR_BUDGET,
    max_active: float = MAX_ACTIVE_WEIGHT,
) -> np.ndarray:
    """Capped active weights for many mixes. ``scores`` is (mixes, names)."""
    weights = np.asarray(index_w, dtype=float)
    total = float(weights.sum())
    if total <= 0:
        return np.zeros_like(scores)
    weights = weights / total
    usable = np.isfinite(scores)
    clean = np.where(usable, scores, 0.0)
    denom = np.where(usable, weights, 0.0)
    denom_sum = denom.sum(axis=1, keepdims=True)
    average = np.divide((clean * denom).sum(axis=1, keepdims=True), denom_sum, where=denom_sum > 0)
    excess = np.where(usable, clean - average, 0.0)
    raw = weights * np.clip(excess, 0.0, None)
    raw_sum = raw.sum(axis=1, keepdims=True)
    funders = np.where(raw <= 0, weights, 0.0)
    funder_sum = funders.sum(axis=1, keepdims=True)
    unit = raw - np.divide(funders, np.maximum(funder_sum, 1e-12)) * raw_sum
    unit = np.where((raw_sum > 0) & (funder_sum > 0), unit, 0.0)
    variance = np.sum((unit @ covariance) * unit, axis=1)
    tracking = np.sqrt(np.clip(variance, 0.0, None)) * np.sqrt(252)
    scale = np.divide(budget, tracking, out=np.zeros_like(tracking), where=tracking > 0)
    extra = np.minimum(scale[:, None] * raw, max_active)
    extra_sum = extra.sum(axis=1, keepdims=True)
    funders = np.where(extra <= 0, weights, 0.0)
    funder_sum = funders.sum(axis=1, keepdims=True)
    active = extra - np.divide(funders, np.maximum(funder_sum, 1e-12)) * extra_sum
    active = np.where((extra_sum > 0) & (funder_sum > 0), active, 0.0)
    fitted = np.sqrt(np.clip(np.sum((active @ covariance) * active, axis=1), 0.0, None)) * np.sqrt(252)
    shrink = np.ones_like(fitted)
    over = (fitted > budget) & (budget > 0) & (fitted > 0)
    shrink[over] = budget / fitted[over]
    return active * shrink[:, None]


def excess_paths(
    prices: pd.DataFrame,
    shares: pd.Series,
    scores: np.ndarray,
    reviews: pd.DatetimeIndex,
) -> np.ndarray:
    """Daily book-minus-index return for each mix.

    ``scores`` has shape (mixes, days, names) and follows ``prices.index``.
    The weight dated on a review earns the next session.
    """
    members = list(prices.columns)
    sessions = prices.index
    returns = prices.pct_change(fill_method=None).to_numpy(dtype=float)
    review_lookup = {pd.Timestamp(day): index for index, day in enumerate(reviews)}
    mixes = scores.shape[0]
    book = np.zeros((mixes, len(members)), dtype=float)
    index_book = np.zeros(len(members), dtype=float)
    excess = np.full((mixes, len(sessions)), np.nan)
    for index, day in enumerate(sessions):
        move = np.nan_to_num(returns[index], nan=0.0)
        if index > 0:
            portfolio = np.nansum(book * move, axis=1)
            benchmark = float(np.nansum(index_book * move))
            excess[:, index] = portfolio - benchmark
        if day in review_lookup:
            base = index_weights(prices.loc[day], shares.reindex(members)).reindex(members).fillna(0.0)
            base_values = base.to_numpy(dtype=float)
            covariance = daily_covariance(prices, day).reindex(index=members, columns=members).fillna(0.0)
            active = _active_batch(base_values, scores[:, index, :], covariance.to_numpy(dtype=float))
            book = base_values + active
            totals = book.sum(axis=1, keepdims=True)
            book = np.divide(book, totals, out=np.zeros_like(book), where=totals > 0)
            index_book = base_values
            continue
        grown = np.clip(book * (1.0 + move), 0.0, None)
        totals = grown.sum(axis=1, keepdims=True)
        book = np.divide(grown, totals, out=book, where=totals > 0)
        index_grown = np.clip(index_book * (1.0 + move), 0.0, None)
        index_total = float(index_grown.sum())
        if index_total > 0:
            index_book = index_grown / index_total
    return excess


def choose_mixes(
    prices: pd.DataFrame,
    shares: pd.Series,
    normalized: pd.DataFrame,
    leaves: pd.DataFrame,
    members: list[str],
) -> pd.DataFrame:
    """One winning mix per leaf, plus the mix used before the tree can speak."""
    sessions = prices.index
    panel = _factor_panel(normalized, members, sessions)
    mixes = grid()
    weights = np.asarray(mixes, dtype=float)
    scores = np.einsum("mf,dnf->mdn", weights, panel)
    reviews = review_sessions(sessions)
    excess = excess_paths(prices[members], shares.reindex(members), scores, reviews)
    month = pd.Series(sessions, index=sessions).dt.to_period("M").dt.to_timestamp()
    flags = leaves.reindex(month.to_numpy()).ffill()
    rows = []
    chosen = {}
    for key in KEYS:
        if key is None:
            mask = flags["high_yield"].isna().to_numpy()
            name = "Before the yield history is long enough"
        else:
            mask = ((flags["high_yield"] == key[0]) & (flags["wide_credit"] == key[1])).fillna(False).to_numpy()
            name = ""
        best_mix = (0.0, 0.0, 0.0, 0.0)
        best_ratio = float("nan")
        for index, mix in enumerate(mixes):
            ratio = information_ratio(excess[index, mask])
            if np.isfinite(ratio) and (not np.isfinite(best_ratio) or ratio > best_ratio):
                best_ratio = ratio
                best_mix = mix
        chosen[key] = best_mix
        rows.append(
            {
                "high_yield": None if key is None else key[0],
                "wide_credit": None if key is None else key[1],
                "name": name,
                "mix": best_mix,
                "information_ratio": best_ratio,
                "days": int(mask.sum()),
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["chosen"] = chosen
    return frame
