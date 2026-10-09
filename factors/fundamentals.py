"""Growth and quality from reported earnings.

Growth is the year-over-year change in earnings per share.
Earnings per share is net income divided by diluted shares.
Quality is return on equity: trailing earnings divided by equity.

Yahoo's statement date is the period end, so each value is held back
until a filing lag has passed, then forward-filled onto price dates.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

QUARTERLY_FILING_LAG_DAYS = 40
ANNUAL_FILING_LAG_DAYS = 60
_YEAR_MATCH_DAYS = 45


def earnings_per_share(net_income: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """EPS = net income / diluted shares. Non-positive share counts are blank."""
    aligned_shares = shares.reindex(index=net_income.index, columns=net_income.columns)
    usable = aligned_shares.where(aligned_shares > 0)
    return net_income / usable


def year_ago_growth(values: pd.DataFrame) -> pd.DataFrame:
    """Change versus the report closest to one year earlier, scaled by its absolute value."""
    ordered = values.sort_index()
    growth = pd.DataFrame(index=ordered.index, columns=ordered.columns, dtype=float)
    dates = list(ordered.index)
    for date in dates:
        target = date - pd.DateOffset(years=1)
        candidates = [
            prev
            for prev in dates
            if prev < date and abs((prev - target).days) <= _YEAR_MATCH_DAYS
        ]
        if not candidates:
            continue
        prior_date = min(candidates, key=lambda prev: abs((prev - target).days))
        prior = ordered.loc[prior_date]
        growth.loc[date] = (ordered.loc[date] - prior) / prior.abs()
    return growth


def annual_return_on_equity(net_income: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    """Full-year earnings over year-end equity. Negative equity is blank."""
    aligned = equity.reindex(index=net_income.index, columns=net_income.columns)
    usable = aligned.where(aligned > 0)
    return net_income / usable


def quarterly_return_on_equity(net_income: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    """Trailing four quarters of earnings over the latest equity."""
    income = net_income.sort_index()
    equity = equity.sort_index()
    roe = pd.DataFrame(index=income.index, columns=income.columns, dtype=float)
    for date in income.index:
        window = income.loc[(income.index > date - pd.Timedelta(days=370)) & (income.index <= date)]
        if len(window) < 4:
            continue
        trailing = window.sum(min_count=4)
        equity_now = equity.reindex(equity.index.union([date])).sort_index().ffill().loc[date]
        roe.loc[date] = trailing / equity_now.where(equity_now > 0)
    return roe


def _dated(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize()
    return out.sort_index()


def _publish(values: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    """Move period-end values forward to the first date they are allowed to be used."""
    if values.empty:
        return values
    published = values.copy()
    published.index = published.index + pd.Timedelta(days=lag_days)
    published = published.sort_index()
    return published.groupby(published.index).last()


def _ffill_daily(events: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(index=index)
    combined = events.sort_index().groupby(level=0).last()
    daily_index = pd.DatetimeIndex(index)
    timeline = combined.reindex(combined.index.union(daily_index)).sort_index().ffill()
    return timeline.reindex(daily_index)


def _merge_published(*frames: pd.DataFrame) -> pd.DataFrame:
    parts = [frame for frame in frames if frame is not None and not frame.empty]
    if not parts:
        return pd.DataFrame()
    merged = pd.concat(parts).sort_index()
    return merged.groupby(level=0).last()


def growth_and_quality(
    quarterly_income: pd.DataFrame,
    quarterly_shares: pd.DataFrame,
    quarterly_equity: pd.DataFrame,
    annual_income: pd.DataFrame,
    annual_shares: pd.DataFrame,
    annual_equity: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily growth and quality. Later reports replace earlier ones after the filing lag."""
    quarterly_eps = earnings_per_share(_dated(quarterly_income), _dated(quarterly_shares))
    annual_eps = earnings_per_share(_dated(annual_income), _dated(annual_shares))
    growth_events = _merge_published(
        _publish(year_ago_growth(annual_eps), ANNUAL_FILING_LAG_DAYS),
        _publish(year_ago_growth(quarterly_eps), QUARTERLY_FILING_LAG_DAYS),
    )
    quality_events = _merge_published(
        _publish(
            annual_return_on_equity(_dated(annual_income), _dated(annual_equity)),
            ANNUAL_FILING_LAG_DAYS,
        ),
        _publish(
            quarterly_return_on_equity(_dated(quarterly_income), _dated(quarterly_equity)),
            QUARTERLY_FILING_LAG_DAYS,
        ),
    )
    growth = _ffill_daily(growth_events, daily_index).replace([float("inf"), float("-inf")], pd.NA)
    quality = _ffill_daily(quality_events, daily_index).replace([float("inf"), float("-inf")], pd.NA)
    return growth, quality


def load_statement_frames(market: str | None = None) -> dict[str, pd.DataFrame] | None:
    """Load saved statement CSVs. None when the download has not been run."""
    from paths import get_paths

    raw_dir, _ = get_paths(market)
    names = {
        "quarterly_income": "net_income_quarterly.csv",
        "quarterly_shares": "shares_quarterly.csv",
        "quarterly_equity": "equity_quarterly.csv",
        "annual_income": "net_income_annual.csv",
        "annual_shares": "shares_annual.csv",
        "annual_equity": "equity_annual.csv",
    }
    frames = {}
    for key, filename in names.items():
        path = Path(raw_dir) / filename
        if not path.exists():
            return None
        frames[key] = pd.read_csv(path, index_col=0, parse_dates=True)
    return frames
