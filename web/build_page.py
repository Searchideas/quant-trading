"""Build the public book page from the frozen leaf mixes.

The mixes are not searched again. A later day uses whichever leaf last
month's 10-year yield and credit spread select. Live weights, config.py,
and the brokerage account are not touched.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from analytics.four_leaf_search import leaf_keys
from analytics.leaf_weight_search import factor_matrix, stitch_scores
from analytics.regime_backtest import load_states_from_pages
from analytics.yield_regime import assign_leaves, load_states
from backtest.engine import BacktestEngine
from strategy.benchmark import Benchmark
from strategy.factor_strategy import (
    load_strategy_data,
    normalize_factors,
    hide_unlisted,
    listed_span,
    select_sector_best_weights,
)

CHOSEN = {
    (False, False): (0.0, 0.0, 0.0, 1.0),
    (False, True): (0.0, 0.0, 0.3, 0.7),
    (True, False): (0.0, 0.6, 0.1, 0.3),
    (True, True): (0.0, 1.0, 0.0, 0.0),
}
LEAF_CODE = {
    (False, False): "0",
    (False, True): "1",
    (True, False): "2",
    (True, True): "3",
}
LEAF_NAME = {
    (False, False): "Low 10-year, tight credit",
    (False, True): "Low 10-year, wide credit",
    (True, False): "High 10-year, tight credit",
    (True, True): "High 10-year, wide credit",
}


def mix_text(mix: tuple[float, float, float, float]) -> str:
    momentum, value, volatility, dividend = mix
    return f"{int(round(momentum * 100))} / {int(round(value * 100))} / {int(round(dividend * 100))} / {int(round(volatility * 100))}"


def load_yield_states() -> pd.DataFrame:
    cache = ROOT / "data" / "us" / "processed" / "yield_states.csv"
    try:
        states = load_states()
    except Exception as exc:
        print(f"FRED download failed ({exc}).")
        if cache.exists():
            states = pd.read_csv(cache, index_col=0, parse_dates=True)
        else:
            states = load_states_from_pages()
    cache.parent.mkdir(parents=True, exist_ok=True)
    states.to_csv(cache)
    return states


def pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    rounded = int(round(value * 100))
    return f"{rounded:+d}%"


def plain_pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{int(round(value * 100))}%"


def _num(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _risk(rets: pd.Series) -> dict[str, float]:
    clean = rets.dropna()
    if len(clean) < 2:
        return {"sharpe": np.nan, "sortino": np.nan, "vol": np.nan, "up": np.nan}
    deviation = float(clean.std(ddof=0))
    downside = float(np.sqrt(np.mean(np.minimum(clean.to_numpy(), 0.0) ** 2)))
    mean = float(clean.mean())
    sharpe = mean / deviation * np.sqrt(252) if deviation else np.nan
    sortino = mean / downside * np.sqrt(252) if downside else np.nan
    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "vol": deviation * np.sqrt(252),
        "up": float((clean > 0).mean()),
    }


def _drawdown(path: pd.Series) -> float:
    if path.empty:
        return np.nan
    return float((path / path.cummax() - 1.0).min())


def _cagr(start: float, end: float, sessions: int) -> float:
    years = sessions / 252
    if years <= 0 or start <= 0:
        return np.nan
    return (end / start) ** (1 / years) - 1


def build_years(equity: pd.Series, spy: pd.Series) -> list[dict]:
    book_daily = equity.pct_change()
    spy_daily = spy.pct_change()
    rows: list[dict] = []
    book_start = float(equity.iloc[0])
    spy_start = float(spy.iloc[0])
    last_year = int(equity.index[-1].year)
    last_month = int(equity.index[-1].month)
    for year in sorted(set(equity.index.year)):
        mask = equity.index.year == year
        book_path = equity.loc[mask]
        spy_path = spy.loc[mask]
        book_end = float(book_path.iloc[-1])
        spy_end = float(spy_path.iloc[-1])
        book_ret = book_end / book_start - 1.0
        spy_ret = spy_end / spy_start - 1.0
        gap_pp = round((book_ret - spy_ret) * 100, 1)
        book_risk = _risk(book_daily.loc[mask])
        spy_risk = _risk(spy_daily.loc[mask])
        partial = year == last_year and last_month < 12
        rows.append(
            {
                "label": f"{year} to date" if partial else str(year),
                "book": pct(book_ret),
                "spy": pct(spy_ret),
                "gap": "0.0 pp" if gap_pp == 0 else f"{gap_pp:+.1f} pp",
                "bookSharpe": _num(book_risk["sharpe"]),
                "spySharpe": _num(spy_risk["sharpe"]),
                "bookSortino": _num(book_risk["sortino"]),
                "spySortino": _num(spy_risk["sortino"]),
                "bookDd": pct(_drawdown(pd.concat([pd.Series([book_start]), book_path]))),
                "spyDd": pct(_drawdown(pd.concat([pd.Series([spy_start]), spy_path]))),
                "bookVol": plain_pct(book_risk["vol"]),
                "spyVol": plain_pct(spy_risk["vol"]),
                "lag": gap_pp < 0,
            }
        )
        book_start = book_end
        spy_start = spy_end
    return rows


def build_scorecard(equity: pd.Series, spy: pd.Series) -> tuple[list[dict], str]:
    book_daily = equity.pct_change().dropna()
    spy_daily = spy.pct_change().dropna()
    book_risk = _risk(book_daily)
    spy_risk = _risk(spy_daily)
    book_cagr = _cagr(float(equity.iloc[0]), float(equity.iloc[-1]), len(book_daily))
    spy_cagr = _cagr(float(spy.iloc[0]), float(spy.iloc[-1]), len(spy_daily))
    book_dd = _drawdown(equity)
    spy_dd = _drawdown(spy)

    def calmar(cagr: float, drawdown: float) -> float:
        if not np.isfinite(cagr) or not np.isfinite(drawdown) or drawdown == 0:
            return np.nan
        return cagr / abs(drawdown)

    rows = [
        {"metric": "Total return", "book": pct(float(equity.iloc[-1]) / float(equity.iloc[0]) - 1), "spy": pct(float(spy.iloc[-1]) / float(spy.iloc[0]) - 1)},
        {"metric": "CAGR", "book": pct(book_cagr), "spy": pct(spy_cagr)},
        {"metric": "Volatility", "book": plain_pct(book_risk["vol"]), "spy": plain_pct(spy_risk["vol"])},
        {"metric": "Sharpe", "book": _num(book_risk["sharpe"]), "spy": _num(spy_risk["sharpe"])},
        {"metric": "Sortino", "book": _num(book_risk["sortino"]), "spy": _num(spy_risk["sortino"])},
        {"metric": "Maximum drawdown", "book": pct(book_dd), "spy": pct(spy_dd)},
        {"metric": "Calmar", "book": _num(calmar(book_cagr, book_dd)), "spy": _num(calmar(spy_cagr, spy_dd))},
        {"metric": "Share of up days", "book": plain_pct(book_risk["up"]), "spy": plain_pct(spy_risk["up"])},
    ]
    book_month = (1 + book_daily).resample("ME").prod() - 1
    spy_month = (1 + spy_daily).resample("ME").prod() - 1
    both = pd.concat([book_month, spy_month], axis=1).dropna()
    ahead = int((both.iloc[:, 0] > both.iloc[:, 1]).sum())
    return rows, f"The book finished ahead of SPY in {ahead} of {len(both)} months."


def build_periods(equity: pd.Series, spy: pd.Series, day_keys: list) -> list[dict]:
    months = equity.index.to_period("M")
    ends = [equity.index[months == period][-1] for period in months.unique()]
    spells: list[dict] = []
    start_level_book = float(equity.iloc[0])
    start_level_spy = float(spy.iloc[0])
    spell_start = ends[0]
    current = day_keys[equity.index.get_loc(ends[0])]
    count = 0
    for day in ends:
        key = day_keys[equity.index.get_loc(day)]
        if key != current and count:
            spells.append(_spell(spell_start, prev, current, start_level_book, start_level_spy, equity, spy, count))
            start_level_book = float(equity.loc[prev])
            start_level_spy = float(spy.loc[prev])
            spell_start = day
            current = key
            count = 0
        prev = day
        count += 1
    spells.append(_spell(spell_start, prev, current, start_level_book, start_level_spy, equity, spy, count))
    return spells


def _spell(start, end, key, book0, spy0, equity, spy, months: int) -> dict:
    book_ret = float(equity.loc[end]) / book0 - 1.0
    spy_ret = float(spy.loc[end]) / spy0 - 1.0
    gap = int(round((book_ret - spy_ret) * 100))
    mix = CHOSEN.get(key, (0.0, 0.6, 0.1, 0.3))
    return {
        "start": start.strftime("%Y-%m"),
        "end": end.strftime("%Y-%m"),
        "leaf": LEAF_NAME.get(key, "Outside the tree"),
        "mix": mix_text(mix),
        "book": pct(book_ret),
        "spy": pct(spy_ret),
        "gap": f"{gap:+d} pp",
        "months": months,
        "lag": book_ret < spy_ret,
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    leaves = assign_leaves(load_yield_states())
    data = load_strategy_data("us")
    normalized, tickers = normalize_factors(
        data["factors_combined"],
        data["categorizations"],
        data["sector_value"],
        data["sector_dividend"],
        data["sector_momentum"],
        data["sector_volatility"],
        data["sector_growth"],
        data["sector_quality"],
    )
    parts = {
        name: factor_matrix(normalized, tickers, name)
        for name in ("momentum", "value", "volatility", "dividend")
    }
    keys = list(leaf_keys(normalized.index, leaves))
    last_key = (True, False)
    for i, key in enumerate(keys):
        if key is None:
            keys[i] = last_key
        else:
            last_key = key
    score = stitch_scores(parts, normalized.index, tickers, keys, CHOSEN)
    prices = data["price_data"].reindex(index=score.index, columns=score.columns)
    score = hide_unlisted(score, prices)
    book = select_sector_best_weights(
        score,
        data["categorizations"],
        data["sector_market_cap"],
        max_sector_weight=1.0,
        top_n_per_sector=2,
    )
    listed = listed_span(prices).reindex(index=book.index, columns=book.columns).fillna(False)
    held_unlisted = book.where(~listed).fillna(0.0)
    if (held_unlisted > 0).any().any():
        raise SystemExit("The book holds a name before it lists or after it is gone.")
    result = BacktestEngine(prices, book, initial_capital=100_000.0).run()
    equity = result["equity"]
    equity = equity[equity.index >= "2017-01-01"]
    spy = (
        Benchmark(
            "SPY",
            start_date="2016-01-01",
            end_date=(equity.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            initial_capital=100_000.0,
        )
        .equity.reindex(equity.index)
        .ffill()
    )
    if spy.isna().any():
        raise SystemExit(f"SPY missing on {int(spy.isna().sum())} sessions")
    costs = result.loc[equity.index, "cost_sgd"]
    turnover = result.loc[equity.index, "turnover"]
    weights = book.reindex(equity.index).fillna(0.0)
    base_book = float(equity.iloc[0])
    base_spy = float(spy.iloc[0])
    day_keys = []
    dates, growth_book, growth_spy, cost_day = [], [], [], []
    traded, leaf_chars, hold_ids, books, seen = [], [], [], [], {}
    for day in equity.index:
        key = keys[normalized.index.get_loc(day)]
        day_keys.append(key)
        row = weights.loc[day]
        held = row[row > 0.001].sort_values(ascending=False)
        text = "|".join(f"{ticker} {weight * 100:.1f}" for ticker, weight in held.items())
        if text not in seen:
            seen[text] = len(books)
            books.append(text)
        dates.append(day.strftime("%Y-%m-%d"))
        growth_book.append(round(100.0 * float(equity.loc[day]) / base_book, 2))
        growth_spy.append(round(100.0 * float(spy.loc[day]) / base_spy, 2))
        cost_day.append(round(float(costs.loc[day]), 2))
        traded.append("1" if float(turnover.loc[day]) > 1e-8 else "0")
        leaf_chars.append(LEAF_CODE.get(key, "2"))
        hold_ids.append(seen[text])

    month_cost: dict[str, float] = {}
    month_rebal: dict[str, int] = {}
    for day, cost, flag in zip(dates, cost_day, traded):
        month = day[:7]
        month_cost[month] = month_cost.get(month, 0.0) + cost
        month_rebal[month] = month_rebal.get(month, 0) + (flag == "1")
    months = list(month_cost)
    periods = build_periods(equity, spy, day_keys)
    years = build_years(equity, spy)
    scorecard, months_ahead = build_scorecard(equity, spy)
    payload = {
        "built": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "dates": dates,
        "book": growth_book,
        "spy": growth_spy,
        "cost": cost_day,
        "traded": "".join(traded),
        "leaf": "".join(leaf_chars),
        "holdId": hold_ids,
        "books": books,
        "months": months,
        "mcost": [round(month_cost[month]) for month in months],
        "mrebal": [month_rebal[month] for month in months],
        "periods": periods,
        "years": years,
        "scorecard": scorecard,
        "monthsAhead": months_ahead,
        "statBook": pct(growth_book[-1] / growth_book[0] - 1.0),
        "statSpy": pct(growth_spy[-1] / growth_spy[0] - 1.0),
        "statCost": f"${float(costs.sum()):,.0f}",
    }
    html = (ROOT / "web" / "template.html").read_text(encoding="utf-8")
    blob = json.dumps(payload).replace("<", "\\u003c")
    html = html.replace("/*@@DATA@@*/null", blob)
    out = ROOT / "web" / "public"
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(html, encoding="utf-8")
    print(dates[0], dates[-1], "days", len(dates), "books", len(books), payload["statBook"], payload["statSpy"])
    for row in years:
        print(
            row["label"], row["book"], row["spy"], row["gap"],
            row["bookSharpe"], row["spySharpe"], row["bookSortino"], row["spySortino"],
            row["bookDd"], row["spyDd"],
        )
    for row in scorecard:
        print(row["metric"], row["book"], row["spy"])
    print(months_ahead)


if __name__ == "__main__":
    main()
