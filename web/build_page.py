"""Build the public book page.

The four mixes are chosen once, on each leaf's own days from 2017 through
2022. From 2023 those mixes stay fixed. A later run appends the new session
onto the saved path and does not replace a saved day. A full rebuild happens
only with --rebuild, or when the saved path is missing. Live weights,
config.py, and the brokerage account are not touched.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from analytics.four_leaf_search import ORDER, leaf_keys
from analytics.four_leaf_sharpe import best_sharpe
from analytics.leaf_weight_search import factor_matrix, grid, gross_returns, score_frame, stitch_scores
from analytics.performance import market_regression
from analytics.regime_backtest import load_states_from_pages
from analytics.yield_regime import assign_leaves, load_states
from backtest.engine import BacktestEngine
from factors.dividend import DividendFactor
from factors.io import load_book_data, load_dividend_data
from factors.momentum import MomentumFactor
from factors.value import ValueFactor
from factors.volatility import VolatilityFactor
from factors import sector_mean
from strategy.benchmark import Benchmark
from strategy.factor_strategy import (
    hide_unlisted,
    listed_span,
    load_strategy_data,
    normalize_factors,
    select_sector_best_weights,
)
from web.page_archive import (
    TRAIN_END,
    TRAIN_START,
    day_weights,
    extend_records,
    holdings_text,
    leaf_text,
    load_categories,
    load_history,
    load_mixes,
    load_prices,
    load_weights,
    ready,
    save_run,
    sector_caps,
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


def _signed_pct(value: float, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value * 100:+.{digits}f}%"


def _model_cells(strategy: pd.Series, market: pd.Series) -> dict[str, str]:
    """Beta, annualized alpha with its t-statistic, and the information ratio."""
    fit = market_regression(strategy, market)
    if not np.isfinite(fit["beta"]):
        return {"beta": "—", "alpha": "—", "ratio": "—"}
    t_stat = fit["alpha_t"]
    alpha = _signed_pct(float(fit["alpha_annualized"]))
    if np.isfinite(t_stat):
        alpha = f"{alpha} (t {t_stat:.1f})"
    ratio = fit["information_ratio"]
    return {
        "beta": _num(float(fit["beta"])),
        "alpha": alpha,
        "ratio": _num(float(ratio)) if np.isfinite(ratio) else "—",
    }


def market_table(strategy: pd.Series, market: pd.Series) -> list[dict]:
    """Full-path market model. Alpha is annualized and its range is 1.96 standard errors."""
    fit = market_regression(strategy, market)
    if not np.isfinite(fit["beta"]):
        return []
    t_stat = fit["alpha_t"]
    t_text = f"{t_stat:.2f}" if np.isfinite(t_stat) else "—"
    ratio = fit["information_ratio"]
    return [
        {"metric": "Beta", "value": f"{fit['beta']:.2f} (se {fit['beta_se']:.2f})"},
        {
            "metric": "Alpha per year",
            "value": (
                f"{_signed_pct(float(fit['alpha_annualized']))} "
                f"(se {abs(float(fit['alpha_annualized_se'])) * 100:.1f}%, t {t_text})"
            ),
        },
        {
            "metric": "Alpha, 95% range",
            "value": f"{_signed_pct(float(fit['alpha_low']))} to {_signed_pct(float(fit['alpha_high']))}",
        },
        {"metric": "Unexplained volatility", "value": plain_pct(float(fit["residual_vol"]))},
        {"metric": "Information ratio", "value": _num(float(ratio)) if np.isfinite(ratio) else "—"},
    ]


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
        fitted = _model_cells(book_daily.loc[mask], spy_daily.loc[mask])
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
                "beta": fitted["beta"],
                "alpha": fitted["alpha"],
                "ratio": fitted["ratio"],
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


def filled_keys(index, leaves) -> list:
    keys = list(leaf_keys(index, leaves))
    last_key = (True, False)
    for i, key in enumerate(keys):
        if key is None:
            keys[i] = last_key
        else:
            last_key = key
    return keys


def choose_train_mixes(parts, prices, categories, caps, index, tickers, keys) -> dict:
    """Highest Sharpe on each leaf, using only 2017 through 2022."""
    mixes = grid(step=10)
    window = index[(index >= TRAIN_START) & (index < TRAIN_END)]
    positions = index.get_indexer(window)
    daily = prices.reindex(index=window, columns=tickers).pct_change(fill_method=None).fillna(0.0)
    library = np.zeros((len(mixes), len(window)))
    window_parts = {name: values[positions] for name, values in parts.items()}
    window_keys = [keys[int(pos)] for pos in positions]
    for i, mix in enumerate(mixes):
        frame = score_frame(window_parts, mix, window, tickers)
        book = select_sector_best_weights(
            frame, categories, caps, max_sector_weight=1.0, top_n_per_sector=2
        )
        library[i] = gross_returns(daily, book)
        if (i + 1) % 40 == 0 or i + 1 == len(mixes):
            print(f"train mix {i + 1}/{len(mixes)}")
    chosen = {}
    for key in ORDER:
        mask = np.array([item == key for item in window_keys], dtype=bool)
        mix, sharpe = best_sharpe(mixes, library, mask)
        chosen[key] = mix
        print(LEAF_NAME[key], mix_text(mix), f"Sharpe {sharpe:.2f}")
    return chosen


def simulate(data, normalized, tickers, keys, mixes):
    parts = {
        name: factor_matrix(normalized, tickers, name)
        for name in ("momentum", "value", "volatility", "dividend")
    }
    score = stitch_scores(parts, normalized.index, tickers, keys, mixes)
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
    equity = equity[equity.index >= TRAIN_START]
    bench = Benchmark(
        "SPY",
        start_date="2016-01-01",
        end_date=(equity.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        initial_capital=100_000.0,
    )
    spy = bench.equity.reindex(equity.index).ffill()
    spy_close = bench.prices.reindex(equity.index).ffill()
    if spy.isna().any() or spy_close.isna().any():
        raise SystemExit(f"SPY missing on {int(spy.isna().sum())} sessions")
    history = pd.DataFrame(index=equity.index)
    history["book"] = equity
    history["spy"] = spy
    history["spy_close"] = spy_close
    history["cost"] = result.loc[equity.index, "cost_sgd"]
    history["turnover"] = result.loc[equity.index, "turnover"]
    history["leaf"] = [leaf_text(keys[normalized.index.get_loc(day)]) for day in equity.index]
    weights = book.reindex(equity.index).fillna(0.0)
    history["holdings"] = [holdings_text(weights.loc[day]) for day in equity.index]
    return history, weights, prices.reindex(equity.index)


def propose_new_book(prices, categories, mixes, leaves, new_days):
    """Scores for the new sessions only. Does not write factor files."""
    value_df = ValueFactor(prices, load_book_data("us")).calculate_value_factor()
    dividend_df = DividendFactor(prices, load_dividend_data("us")).calculate_dividend_factor()
    momentum_df = MomentumFactor(prices).calculate_momentum()
    volatility_df = VolatilityFactor(prices).calculate_volatility()
    combined = pd.concat(
        [
            dividend_df.add_suffix("_dividend"),
            value_df.add_suffix("_value"),
            momentum_df.add_suffix("_momentum"),
            volatility_df.add_suffix("_volatility"),
        ],
        axis=1,
    )
    sector_frames = {}
    for name, frame in (
        ("value", value_df),
        ("dividend", dividend_df),
        ("momentum", momentum_df),
        ("volatility", volatility_df),
    ):
        mean_df, std_df = sector_mean(frame, categories)
        sector_frames[name] = pd.concat(
            {f"{sector}_mean": mean_df[sector] for sector in mean_df.columns}
            | {f"{sector}_std": std_df[sector] for sector in std_df.columns},
            axis=1,
        )
    normalized, tickers = normalize_factors(
        combined,
        categories,
        sector_frames["value"],
        sector_frames["dividend"],
        sector_frames["momentum"],
        sector_frames["volatility"],
    )
    parts = {
        name: factor_matrix(normalized, tickers, name)
        for name in ("momentum", "value", "volatility", "dividend")
    }
    keys = filled_keys(normalized.index, leaves)
    score = hide_unlisted(
        stitch_scores(parts, normalized.index, tickers, keys, mixes),
        prices.reindex(columns=tickers),
    )
    caps = sector_caps(categories)
    rows = []
    leaves_out = []
    for day in new_days:
        if day not in score.index:
            rows.append(pd.Series(0.0, index=score.columns))
            leaves_out.append("")
            continue
        loc = score.index.get_loc(day)
        rows.append(day_weights(score.iloc[loc], categories, caps))
        leaves_out.append(leaf_text(keys[loc]))
    return pd.DataFrame(rows, index=new_days), pd.Series(leaves_out, index=new_days)


def render(history: pd.DataFrame) -> None:
    global CHOSEN
    CHOSEN = load_mixes()
    equity = history["book"].astype(float)
    spy = history["spy"].astype(float)
    costs = history["cost"].astype(float)
    turnover = history["turnover"].astype(float)
    day_keys = []
    base_book = float(equity.iloc[0])
    base_spy = float(spy.iloc[0])
    dates, growth_book, growth_spy, cost_day = [], [], [], []
    traded, leaf_chars, hold_ids, books, seen = [], [], [], [], {}
    for day in equity.index:
        high, wide = str(history.loc[day, "leaf"]).split(",")
        key = (high == "1", wide == "1")
        day_keys.append(key)
        text = str(history.loc[day, "holdings"])
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
    model = market_table(equity.pct_change(), spy.pct_change())
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
        "market": model,
        "monthsAhead": months_ahead,
        "mixes": [
            mix_text(CHOSEN[key])
            for key in ((False, False), (False, True), (True, False), (True, True))
        ],
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


def rebuild() -> None:
    warnings.filterwarnings("ignore")
    print("Rebuilding the saved path. Later runs append.")
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
    keys = filled_keys(normalized.index, leaves)
    mixes = choose_train_mixes(
        parts,
        data["price_data"],
        data["categorizations"],
        data["sector_market_cap"],
        normalized.index,
        tickers,
        keys,
    )
    history, weights, prices = simulate(data, normalized, tickers, keys, mixes)
    save_run(history, weights, prices, data["categorizations"], mixes)
    render(history)


def extend() -> bool:
    """Add sessions after the saved path. Returns True when a day was added."""
    warnings.filterwarnings("ignore")
    from download_data import download_yahoo_prices

    history = load_history()
    weights = load_weights()
    prices = load_prices()
    categories = load_categories()
    mixes = load_mixes()
    last = history.index[-1]
    symbols = list(prices.columns)
    fresh = download_yahoo_prices(symbols, last.strftime("%Y-%m-%d"))
    if fresh is None or fresh.empty:
        print(f"No new prices after {last.date()}.")
        return False
    fresh.index = pd.to_datetime(fresh.index)
    if getattr(fresh.index, "tz", None) is not None:
        fresh.index = fresh.index.tz_convert(None)
    fresh.index = fresh.index.normalize()
    added = fresh.index[fresh.index > last]
    if len(added) == 0:
        print(f"No new session after {last.date()}.")
        return False
    bench = Benchmark(
        "SPY",
        start_date=last.strftime("%Y-%m-%d"),
        end_date=(added.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        initial_capital=100_000.0,
    )
    combined = pd.concat([prices, fresh.reindex(columns=prices.columns).loc[added]])
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    leaves = assign_leaves(load_yield_states())
    new_weights, new_leaves = propose_new_book(combined, categories, mixes, leaves, added)
    history, weights, prices = extend_records(
        history, weights, prices, new_weights, fresh, bench.prices, new_leaves
    )
    save_run(history, weights, prices, categories, mixes)
    render(history)
    print(f"Added {len(added)} session(s) through {history.index[-1].date()}.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Choose the 2017-2022 mixes again and replace the saved path.",
    )
    args = parser.parse_args()
    if args.rebuild or not ready():
        rebuild()
        return
    try:
        extend()
    except Exception as exc:
        print(f"Could not add a session ({exc}). The saved path is unchanged.")


if __name__ == "__main__":
    main()
