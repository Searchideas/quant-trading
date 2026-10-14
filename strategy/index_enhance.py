"""Straits Times Index enhancement in the NBIM style.

The index is the portfolio. The factor score only adds weight to names that
rank above the index, and those additions are funded by trimming the other
index names. The additions are then scaled so the predicted tracking error
is inside 1.25 percentage points a year.

The share count is today's market cap divided by the latest price. Official
historical free-float weights are not in this project. Membership is the
30 names in the Straits Times Index as of October 2026, used at every past
review.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The 30 names in the Straits Times Index as of October 2026.
STI_MEMBERS = (
    "D05.SI",  # DBS
    "O39.SI",  # OCBC
    "U11.SI",  # UOB
    "Z74.SI",  # Singtel
    "S68.SI",  # Singapore Exchange
    "S63.SI",  # ST Engineering
    "BN4.SI",  # Keppel
    "C38U.SI",  # CapitaLand Integrated Commercial Trust
    "J36.SI",  # Jardine Matheson
    "BS6.SI",  # Yangzijiang Shipbuilding
    "C6L.SI",  # Singapore Airlines
    "H78.SI",  # Hongkong Land
    "A17U.SI",  # CapitaLand Ascendas REIT
    "F34.SI",  # Wilmar
    "9CI.SI",  # CapitaLand Investment
    "U96.SI",  # Sembcorp Industries
    "AJBU.SI",  # Keppel DC REIT
    "V03.SI",  # Venture
    "5E2.SI",  # Seatrium
    "ME8U.SI",  # Mapletree Industrial Trust
    "U14.SI",  # UOL
    "M44U.SI",  # Mapletree Logistics Trust
    "G13.SI",  # Genting Singapore
    "S58.SI",  # SATS
    "C09.SI",  # City Developments
    "Y92.SI",  # Thai Beverage
    "N2IU.SI",  # Mapletree Pan Asia Commercial Trust
    "J69U.SI",  # Frasers Centrepoint Trust
    "BUOU.SI",  # Frasers Logistics & Commercial Trust
    "D01.SI",  # DFI Retail
)

TRACKING_ERROR_BUDGET = 0.0125
MAX_ACTIVE_WEIGHT = 0.01
COVARIANCE_WINDOW = 252


def third_friday(year: int, month: int) -> pd.Timestamp:
    """Third Friday of a calendar month."""
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (4 - int(first.weekday())) % 7
    return first + pd.Timedelta(days=offset + 14)


def review_sessions(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading session on or before each March and September third Friday."""
    sessions = pd.DatetimeIndex(pd.to_datetime(sessions)).sort_values().unique()
    chosen = []
    if len(sessions) == 0:
        return pd.DatetimeIndex(chosen)
    for year in range(int(sessions[0].year), int(sessions[-1].year) + 1):
        for month in (3, 9):
            friday = third_friday(year, month)
            eligible = sessions[sessions <= friday]
            if len(eligible) == 0 or eligible[-1] < sessions[0]:
                continue
            if friday < sessions[0]:
                continue
            chosen.append(eligible[-1])
    return pd.DatetimeIndex(chosen)


def share_proxy(latest_prices: pd.Series, market_cap: pd.Series) -> pd.Series:
    """Today's market cap divided by the latest price.

    This is a current share-count proxy, not a historical free-float file.
    """
    prices = latest_prices.replace(0, np.nan)
    shares = market_cap.reindex(prices.index) / prices
    return shares.replace([np.inf, -np.inf], np.nan)


def index_weights(prices: pd.Series, shares: pd.Series) -> pd.Series:
    """Free-float proxy weights from one session's prices and a share count."""
    value = prices.astype(float) * shares.reindex(prices.index).astype(float)
    value = value.where(prices.astype(float) > 0)
    value = value.where(value > 0)
    total = float(value.sum(skipna=True))
    if not np.isfinite(total) or total <= 0:
        return pd.Series(0.0, index=prices.index)
    weights = value / total
    return weights.fillna(0.0)


def selective_active(
    index_w: pd.Series,
    score: pd.Series,
    scale: float,
    max_active: float,
) -> pd.Series:
    """Overweight names above the index-average score, funded by the rest.

    Active weights sum to zero. A name at or below the average is not increased.
    """
    weights = index_w.astype(float)
    total = float(weights.sum())
    if total <= 0:
        return pd.Series(0.0, index=weights.index)
    weights = weights / total
    scores = score.reindex(weights.index).astype(float)
    usable = scores.notna() & weights.gt(0)
    active = pd.Series(0.0, index=weights.index, dtype=float)
    if int(usable.sum()) < 2 or scale == 0:
        return active
    average = float(np.average(scores[usable], weights=weights[usable]))
    excess = (scores - average).where(usable, 0.0).fillna(0.0)
    extra = (scale * weights * excess.clip(lower=0.0)).clip(upper=max_active)
    budget = float(extra.sum())
    funders = weights.where(extra <= 0, 0.0)
    if budget <= 0 or float(funders.sum()) <= 0:
        return active
    active = extra - funders / float(funders.sum()) * budget
    return active


def tracking_error(active: pd.Series, daily_cov: pd.DataFrame) -> float:
    """Annualized tracking error from a daily covariance matrix."""
    names = [name for name in active.index if name in daily_cov.columns and name in daily_cov.index]
    if not names:
        return 0.0
    vector = active.reindex(names).fillna(0.0).to_numpy(dtype=float)
    matrix = daily_cov.loc[names, names].fillna(0.0).to_numpy(dtype=float)
    variance = float(vector @ matrix @ vector)
    if not np.isfinite(variance) or variance <= 0:
        return 0.0
    return float(np.sqrt(variance) * np.sqrt(252))


def fit_active(
    index_w: pd.Series,
    score: pd.Series,
    daily_cov: pd.DataFrame,
    budget: float = TRACKING_ERROR_BUDGET,
    max_active: float = MAX_ACTIVE_WEIGHT,
) -> pd.Series:
    """Scale the overweight until predicted tracking error is inside the budget."""
    unit = selective_active(index_w, score, scale=1.0, max_active=np.inf)
    raw = tracking_error(unit, daily_cov)
    scale = 0.0 if raw <= 0 else budget / raw
    active = selective_active(index_w, score, scale=scale, max_active=max_active)
    fitted = tracking_error(active, daily_cov)
    if fitted > budget > 0:
        active = active * (budget / fitted)
    return active


def daily_covariance(prices: pd.DataFrame, end: pd.Timestamp, window: int = COVARIANCE_WINDOW) -> pd.DataFrame:
    """Covariance of daily returns known on ``end``. Later sessions are excluded."""
    history = prices.loc[:end].tail(window + 1)
    returns = history.pct_change(fill_method=None)
    return returns.cov()


def drift(start: pd.Series, prices: pd.DataFrame) -> pd.DataFrame:
    """Carry weights forward with each session's price move, renormalized."""
    columns = list(prices.columns)
    weights = start.reindex(columns).fillna(0.0).astype(float)
    if float(weights.sum()) > 0:
        weights = weights / float(weights.sum())
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    rows = []
    for day in prices.index:
        rows.append(weights.to_numpy(dtype=float))
        grown = weights * (1.0 + returns.loc[day].reindex(columns).fillna(0.0))
        grown = grown.clip(lower=0.0)
        total = float(grown.sum())
        weights = grown / total if total > 0 else weights
    return pd.DataFrame(rows, index=prices.index, columns=columns)


def build_books(
    prices: pd.DataFrame,
    shares: pd.Series,
    scores: pd.DataFrame,
    budget: float = TRACKING_ERROR_BUDGET,
    max_active: float = MAX_ACTIVE_WEIGHT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Index book and enhanced book, reset on each STI review session.

    The weight dated on a session earns the next session's return. The review
    close decides the book; it is applied on the following session.
    """
    members = [name for name in shares.index if name in prices.columns]
    prices = prices[members].sort_index()
    reviews = review_sessions(prices.index)
    index_rows = []
    port_rows = []
    review_rows = []
    held_index = pd.Series(0.0, index=members)
    held_port = pd.Series(0.0, index=members)
    review_lookup = set(reviews)
    for session in prices.index:
        if session in review_lookup:
            current_prices = prices.loc[session]
            base = index_weights(current_prices, shares.reindex(members))
            cov = daily_covariance(prices[members], session)
            score = scores.reindex(columns=members).loc[session] if session in scores.index else pd.Series(dtype=float)
            active = fit_active(base, score, cov, budget=budget, max_active=max_active)
            held_index = base.reindex(members).fillna(0.0)
            held_port = (held_index + active.reindex(members).fillna(0.0)).clip(lower=0.0)
            if float(held_port.sum()) > 0:
                held_port = held_port / float(held_port.sum())
            review_rows.append(
                {
                    "date": session,
                    "predicted_tracking_error": tracking_error(held_port - held_index, cov),
                    "active_gross": float((held_port - held_index).abs().sum() / 2),
                }
            )
        index_rows.append(held_index.to_numpy(dtype=float))
        port_rows.append(held_port.to_numpy(dtype=float))
    index_book = pd.DataFrame(index_rows, index=prices.index, columns=members)
    port_book = pd.DataFrame(port_rows, index=prices.index, columns=members)
    index_book = _drift_between(index_book, prices, reviews)
    port_book = _drift_between(port_book, prices, reviews)
    reviews_frame = pd.DataFrame(review_rows)
    return index_book, port_book, reviews_frame


def _drift_between(book: pd.DataFrame, prices: pd.DataFrame, reviews: pd.DatetimeIndex) -> pd.DataFrame:
    """Replace post-review rows with weights drifted from the review close."""
    if book.empty:
        return book
    out = book.copy()
    review_set = set(pd.DatetimeIndex(reviews))
    returns = prices.reindex(columns=book.columns).pct_change(fill_method=None)
    current = book.iloc[0].astype(float).copy()
    for session in book.index:
        if session in review_set or float(current.sum()) == 0:
            current = book.loc[session].astype(float).copy()
            out.loc[session] = current
            continue
        grown = current * (1.0 + returns.loc[session].reindex(current.index).fillna(0.0))
        grown = grown.clip(lower=0.0)
        total = float(grown.sum())
        current = grown / total if total > 0 else current
        out.loc[session] = current
    return out


def result_frame(index_book: pd.DataFrame, port_book: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Daily index and portfolio returns.

    The book stored on a session was set at the previous close, and it earns
    this session's close-to-close move. The decision close itself is not earned.
    """
    returns = prices.reindex(columns=index_book.columns).pct_change(fill_method=None)
    held_index = index_book.shift(1).fillna(0.0)
    held_port = port_book.shift(1).fillna(0.0)
    index_return = (held_index * returns).sum(axis=1, min_count=1)
    port_return = (held_port * returns).sum(axis=1, min_count=1)
    frame = pd.DataFrame(
        {
            "index_return": index_return,
            "portfolio_return": port_return,
        }
    )
    frame["excess_return"] = frame["portfolio_return"] - frame["index_return"]
    usable = frame[["index_return", "portfolio_return"]].notna().all(axis=1)
    frame = frame.loc[usable]
    frame["index_equity"] = (1.0 + frame["index_return"].fillna(0.0)).cumprod()
    frame["portfolio_equity"] = (1.0 + frame["portfolio_return"].fillna(0.0)).cumprod()
    return frame


def summarize(results: pd.DataFrame, reviews: pd.DataFrame) -> dict[str, float]:
    """Excess return, realized tracking error, and information ratio."""
    excess = results["excess_return"].dropna()
    deviation = float(excess.std(ddof=0)) if len(excess) else np.nan
    tracking = deviation * np.sqrt(252) if np.isfinite(deviation) else np.nan
    mean = float(excess.mean()) if len(excess) else np.nan
    information = mean / deviation * np.sqrt(252) if deviation else np.nan
    index_total = float(results["index_equity"].iloc[-1] - 1) if len(results) else np.nan
    port_total = float(results["portfolio_equity"].iloc[-1] - 1) if len(results) else np.nan
    return {
        "index_return": index_total,
        "portfolio_return": port_total,
        "excess_return": port_total - index_total if np.isfinite(port_total) else np.nan,
        "realized_tracking_error": tracking,
        "information_ratio": information,
        "reviews": float(len(reviews)),
    }


def singapore_leaves(root: Path) -> pd.DataFrame:
    """MAS 10-year yield and the high-yield minus investment-grade spread."""
    from analytics.sg_leaf import (
        assign_leaves,
        credit_spread,
        distribution_yield,
        parse_sgs_10y,
        tiger_bond_prices,
    )

    raw = root / "data" / "sg" / "raw"
    yield_10y = parse_sgs_10y((raw / "sgs_10y.csv").read_text(encoding="utf-8"))
    bond_prices = tiger_bond_prices(raw / "bond_etf_bars.csv")
    bond_dividends = pd.read_csv(raw / "bond_etf_dividends.csv", index_col=0, parse_dates=True)
    yields = distribution_yield(bond_prices, bond_dividends)
    return assign_leaves(yield_10y, credit_spread(yields))


def run_singapore(root: Path | None = None) -> dict[str, pd.DataFrame | dict]:
    """Build the enhancement from the Singapore files already on disk."""
    from analytics.sg_leaf import LEAF_NAME, blended_scores, leaf_on, mix_on, mix_text
    from analytics.sg_leaf_search import choose_mixes

    root = ROOT if root is None else Path(root)
    raw = root / "data" / "sg" / "raw"
    processed = root / "data" / "sg" / "processed"
    prices = pd.read_csv(raw / "price_data.csv", index_col=0, parse_dates=True)
    categories = pd.read_csv(raw / "categorizations_data.csv", index_col=0)
    normalized = pd.read_csv(processed / "factors_normalized.csv", index_col=0, parse_dates=True)
    members = [name for name in STI_MEMBERS if name in prices.columns]
    leaves = singapore_leaves(root)
    latest = prices[members].ffill().iloc[-1]
    shares = share_proxy(latest, categories["market_cap"])
    search = choose_mixes(prices[members], shares, normalized, leaves, members)
    search["name"] = [
        "Before the yield history is long enough" if pd.isna(row.high_yield) else LEAF_NAME[(bool(row.high_yield), bool(row.wide_credit))]
        for row in search.itertuples(index=False)
    ]
    chosen = search.attrs["chosen"]
    scores = blended_scores(normalized, prices.index, members, leaves, chosen)
    index_book, port_book, reviews = build_books(prices[members], shares.reindex(members), scores)
    if not reviews.empty:
        reviews["leaf"] = [leaf_on(day, leaves) for day in reviews["date"]]
        reviews["mix"] = [mix_text(mix_on(day, leaves, chosen)) for day in reviews["date"]]
    results = result_frame(index_book, port_book, prices[members])
    summary = summarize(results, reviews)
    processed.mkdir(parents=True, exist_ok=True)
    results.to_csv(processed / "sti_enhance_daily.csv")
    reviews.to_csv(processed / "sti_enhance_reviews.csv", index=False)
    latest_port = port_book.iloc[-1]
    latest_index = index_book.iloc[-1]
    latest_active = (latest_port - latest_index).sort_values(ascending=False)
    latest_active.to_csv(processed / "sti_enhance_active.csv", header=["active_weight"])
    saved = search.drop(columns=["mix"]).copy()
    saved["momentum"] = [mix[0] for mix in search["mix"]]
    saved["value"] = [mix[1] for mix in search["mix"]]
    saved["low_volatility"] = [mix[2] for mix in search["mix"]]
    saved["dividend"] = [mix[3] for mix in search["mix"]]
    saved.to_csv(processed / "sti_leaf_mixes.csv", index=False)
    return {
        "summary": summary,
        "results": results,
        "reviews": reviews,
        "active": latest_active,
        "index": latest_index,
        "portfolio": latest_port,
        "index_book": index_book,
        "portfolio_book": port_book,
        "leaves": leaves,
        "search": search,
    }


def main() -> None:
    payload = run_singapore()
    summary = payload["summary"]
    print("STI enhancement, NBIM-style tilt")
    print(f"Index names in the file: {len(STI_MEMBERS)}")
    print(f"Index total return: {summary['index_return']:+.2%}")
    print(f"Enhanced total return: {summary['portfolio_return']:+.2%}")
    print(f"Excess return: {summary['excess_return']:+.2%}")
    print(f"Realized tracking error: {summary['realized_tracking_error']:.2%}")
    print(f"Information ratio: {summary['information_ratio']:.2f}")
    print(f"Reviews: {int(summary['reviews'])}")
    if not payload["reviews"].empty and "leaf" in payload["reviews"].columns:
        last = payload["reviews"].iloc[-1]
        print(f"Latest review {pd.Timestamp(last['date']).date()}: {last['leaf']}, mix {last['mix']}")
    print("Largest overweights at the latest close:")
    print(payload["active"].head(5).map(lambda value: f"{value:+.2%}").to_string())


if __name__ == "__main__":
    main()
