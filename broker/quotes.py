"""Tiger daily closes, with Yahoo used only to fill gaps and flag differences."""

from __future__ import annotations

import time

import pandas as pd

from broker.plan import tiger_to_yahoo, yahoo_to_tiger

_BATCH = 50
_BAR_LIMIT = 1200
_MISMATCH = 0.01


def bars_to_closes(bars: pd.DataFrame, market: str) -> pd.DataFrame:
    """Pivot Tiger bar rows into a date x Yahoo-ticker close table."""
    if bars is None or bars.empty or "close" not in bars.columns:
        return pd.DataFrame()
    frame = bars.dropna(subset=["close"]).copy()
    if frame.empty:
        return pd.DataFrame()
    tz = "America/New_York" if market == "us" else "Asia/Singapore"
    frame["date"] = (
        pd.to_datetime(frame["time"], unit="ms", utc=True)
        .dt.tz_convert(tz)
        .dt.normalize()
        .dt.tz_localize(None)
    )
    frame["yahoo"] = frame["symbol"].map(lambda symbol: tiger_to_yahoo(str(symbol), market))
    close = frame.pivot_table(index="date", columns="yahoo", values="close", aggfunc="last")
    close.index.name = "Date"
    close.columns.name = None
    return close.sort_index()


def dividends_to_frame(dividends: pd.DataFrame, market: str) -> pd.DataFrame:
    """Pivot Tiger dividend rows into ex-date x Yahoo-ticker amounts."""
    if dividends is None or dividends.empty:
        return pd.DataFrame()
    if "execute_date" not in dividends.columns or "amount" not in dividends.columns:
        return pd.DataFrame()
    frame = dividends.dropna(subset=["execute_date", "amount"]).copy()
    if frame.empty:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["execute_date"]).dt.tz_localize(None).dt.normalize()
    frame["yahoo"] = frame["symbol"].map(lambda symbol: tiger_to_yahoo(str(symbol), market))
    wide = frame.pivot_table(index="date", columns="yahoo", values="amount", aggfunc="sum")
    wide.index.name = "Date"
    wide.columns.name = None
    return wide.sort_index()


def combine_closes(tiger: pd.DataFrame, yahoo: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Keep Tiger prices. Use Yahoo only for symbols Tiger did not return."""
    tiger = pd.DataFrame() if tiger is None else tiger.copy()
    yahoo = pd.DataFrame() if yahoo is None else yahoo.copy()
    if tiger.empty:
        yahoo = yahoo.dropna(axis=1, how="all").dropna(axis=0, how="all")
        yahoo.index.name = "Date"
        return yahoo, list(yahoo.columns)

    filled: list[str] = []
    out = tiger.copy()
    for column in yahoo.columns:
        if column not in out.columns or out[column].dropna().empty:
            out[column] = yahoo[column]
            filled.append(str(column))
    out = out.sort_index().dropna(axis=1, how="all").dropna(axis=0, how="all")
    out.index.name = "Date"
    return out, filled


def close_mismatches(
    tiger: pd.DataFrame,
    yahoo: pd.DataFrame,
    threshold: float = _MISMATCH,
) -> pd.DataFrame:
    """Last overlapping close where Tiger and Yahoo differ by more than threshold."""
    if tiger is None or yahoo is None or tiger.empty or yahoo.empty:
        return pd.DataFrame(columns=["ticker", "date", "tiger", "yahoo", "gap"])
    rows = []
    for ticker in tiger.columns.intersection(yahoo.columns):
        pair = pd.concat(
            [tiger[ticker], yahoo[ticker]], axis=1, keys=["tiger", "yahoo"]
        ).dropna()
        if pair.empty:
            continue
        tiger_px = float(pair.iloc[-1]["tiger"])
        yahoo_px = float(pair.iloc[-1]["yahoo"])
        if yahoo_px == 0:
            continue
        gap = abs(tiger_px - yahoo_px) / abs(yahoo_px)
        if gap > threshold:
            rows.append(
                {
                    "ticker": ticker,
                    "date": pair.index[-1],
                    "tiger": tiger_px,
                    "yahoo": yahoo_px,
                    "gap": gap,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "tiger", "yahoo", "gap"])
    return pd.DataFrame(rows).sort_values("gap", ascending=False).reset_index(drop=True)


def _quote_client():
    from tigeropen.quote.quote_client import QuoteClient

    from broker.tiger_client import load_client_config

    return QuoteClient(load_client_config())


def _rate_limited(exc: Exception) -> bool:
    text = str(exc).lower()
    return "rate limit" in text or "current limiting" in text


def _call_with_retry(action, label: str):
    """Tiger allows 60 kline requests a minute. Wait and retry when that cap hits."""
    for attempt in range(4):
        try:
            return action()
        except Exception as exc:
            if not _rate_limited(exc) or attempt == 3:
                raise
            print(f"Tiger rate limit on {label}. Waiting 60s.")
            time.sleep(60)
    return None


def _fetch_bars(client, tiger_symbols: list[str], start: str, end: str) -> pd.DataFrame:
    from tigeropen.common.consts import BarPeriod, QuoteRight

    span_days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    frames: list[pd.DataFrame] = []
    if span_days <= 1600:
        for offset in range(0, len(tiger_symbols), _BATCH):
            batch = tiger_symbols[offset : offset + _BATCH]

            def _batch(batch=batch):
                return client.get_bars(
                    batch,
                    period=BarPeriod.DAY,
                    begin_time=start,
                    end_time=end,
                    limit=_BAR_LIMIT,
                    right=QuoteRight.BR,
                )

            bars = _call_with_retry(_batch, ",".join(batch[:3]))
            if bars is not None and not bars.empty:
                frames.append(bars)
            time.sleep(1.1)
    else:
        for symbol in tiger_symbols:
            def _pages(symbol=symbol):
                return client.get_bars_by_page(
                    symbol,
                    period=BarPeriod.DAY,
                    begin_time=start,
                    end_time=end,
                    total=4000,
                    page_size=1000,
                    right=QuoteRight.BR,
                    time_interval=2,
                )

            try:
                bars = _call_with_retry(_pages, symbol)
            except Exception as exc:
                print(f"Tiger prices skipped {symbol} ({exc}).")
                continue
            if bars is not None and not bars.empty:
                frames.append(bars)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def download_tiger_closes(
    symbols: list[str],
    start_date: str,
    market: str,
) -> pd.DataFrame:
    """Daily closes from Tiger. Empty frame when credentials or the request fail."""
    import os

    from broker.tiger_client import props_file

    if props_file() is None and not os.environ.get("TIGER_ID"):
        print("Tiger config not found. Yahoo will supply prices.")
        return pd.DataFrame()
    end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tiger_symbols = [yahoo_to_tiger(symbol, market) for symbol in symbols]
    try:
        client = _quote_client()
        bars = _fetch_bars(client, tiger_symbols, start_date, end)
    except (SystemExit, Exception) as exc:
        print(f"Tiger price download failed ({exc}). Yahoo will supply prices.")
        return pd.DataFrame()
    closes = bars_to_closes(bars, market)
    if closes.empty:
        print("Tiger returned no prices. Yahoo will supply prices.")
    return closes


def download_tiger_dividends(
    symbols: list[str],
    start_date: str,
    market: str,
) -> pd.DataFrame:
    """Ex-date dividend amounts from Tiger. Empty when the request cannot run."""
    import os

    from broker.tiger_client import props_file

    if props_file() is None and not os.environ.get("TIGER_ID"):
        print("Tiger config not found. Yahoo will supply dividends.")
        return pd.DataFrame()

    begin = (pd.Timestamp(start_date) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tiger_symbols = [yahoo_to_tiger(symbol, market) for symbol in symbols]
    market_code = "US" if market == "us" else "SG"
    timezone = "US/Eastern" if market == "us" else "Asia/Singapore"
    frames: list[pd.DataFrame] = []
    try:
        client = _quote_client()
        for offset in range(0, len(tiger_symbols), 20):
            batch = tiger_symbols[offset : offset + 20]
            rows = client.get_corporate_dividend(
                batch,
                market_code,
                begin,
                end,
                timezone=timezone,
            )
            if rows is not None and not getattr(rows, "empty", True):
                frames.append(rows)
            time.sleep(0.2)
    except (SystemExit, Exception) as exc:
        print(f"Tiger dividend download failed ({exc}). Yahoo will supply dividends.")
        return pd.DataFrame()
    if not frames:
        print("Tiger returned no dividends. Yahoo will supply dividends.")
        return pd.DataFrame()
    return dividends_to_frame(pd.concat(frames, ignore_index=True), market)
