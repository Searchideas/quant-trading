"""Write the Singapore index-enhancement page.

The book is the Straits Times proxy plus a capped score tilt. The factor mix
follows the same four leaves as the US page. Open web/public/sti.html.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from analytics.sg_leaf import (
    distribution_yield,
    month_facts,
    parse_sgs_10y,
    tiger_bond_prices,
)
from analytics.sg_leaf import mix_text
from strategy.index_enhance import run_singapore

NAMES = {
    "D05.SI": "DBS",
    "O39.SI": "OCBC",
    "U11.SI": "UOB",
    "Z74.SI": "Singtel",
    "S68.SI": "SGX",
    "S63.SI": "ST Engineering",
    "BN4.SI": "Keppel",
    "C38U.SI": "CICT",
    "J36.SI": "Jardine Matheson",
    "BS6.SI": "Yangzijiang",
    "C6L.SI": "Singapore Airlines",
    "H78.SI": "Hongkong Land",
    "A17U.SI": "CapitaLand Ascendas REIT",
    "F34.SI": "Wilmar",
    "9CI.SI": "CapitaLand Investment",
    "U96.SI": "Sembcorp",
    "AJBU.SI": "Keppel DC REIT",
    "V03.SI": "Venture",
    "5E2.SI": "Seatrium",
    "ME8U.SI": "Mapletree Industrial",
    "U14.SI": "UOL",
    "M44U.SI": "Mapletree Logistics",
    "G13.SI": "Genting",
    "S58.SI": "SATS",
    "C09.SI": "City Developments",
    "Y92.SI": "Thai Beverage",
    "N2IU.SI": "Mapletree Pan Asia",
    "J69U.SI": "Frasers Centrepoint",
    "BUOU.SI": "Frasers Logistics",
    "D01.SI": "DFI Retail",
}


def _facts(root: Path) -> pd.DataFrame:
    raw = root / "data" / "sg" / "raw"
    yield_10y = parse_sgs_10y((raw / "sgs_10y.csv").read_text(encoding="utf-8"))
    prices = tiger_bond_prices(raw / "bond_etf_bars.csv")
    dividends = pd.read_csv(raw / "bond_etf_dividends.csv", index_col=0, parse_dates=True)
    yields = distribution_yield(prices, dividends)
    return month_facts(yield_10y, yields)


def _mix_words(mix: tuple[float, float, float, float]) -> str:
    momentum, value, volatility, dividend = mix
    parts = []
    for label, weight in (
        ("momentum", momentum),
        ("value", value),
        ("dividend", dividend),
        ("low volatility", volatility),
    ):
        if weight:
            parts.append(f"{int(round(weight * 100))}% {label}")
    return ", ".join(parts) if parts else "no score, so the book matches the index"


def _mix_payload(search: pd.DataFrame) -> tuple[list[dict], dict]:
    mixes = []
    waiting = {}
    for record in search.to_dict("records"):
        mix = record["mix"]
        ratio = record["information_ratio"]
        item = {
            "name": record["name"],
            "words": _mix_words(mix),
            "mix": mix_text(mix),
            "ir": None if pd.isna(ratio) else round(float(ratio), 2),
            "days": int(record["days"]),
        }
        if pd.isna(record["high_yield"]):
            waiting = item
            continue
        item["id"] = int(bool(record["high_yield"])) * 2 + int(bool(record["wide_credit"]))
        mixes.append(item)
    mixes.sort(key=lambda item: item["id"])
    return mixes, waiting


def _year_rows(results: pd.DataFrame) -> list[dict]:
    rows = []
    for year, group in results.groupby(results.index.year):
        index_equity = (1.0 + group["index_return"].fillna(0.0)).prod() - 1.0
        port_equity = (1.0 + group["portfolio_return"].fillna(0.0)).prod() - 1.0
        excess = group["excess_return"].dropna()
        deviation = float(excess.std(ddof=0)) if len(excess) > 1 else float("nan")
        tracking = deviation * (252 ** 0.5) if deviation == deviation else float("nan")
        rows.append(
            {
                "year": int(year),
                "book": round(port_equity, 6),
                "sti": round(index_equity, 6),
                "excess": round(port_equity - index_equity, 6),
                "te": round(tracking, 6),
            }
        )
    return rows


def _page_data(root: Path) -> dict:
    payload = run_singapore(root)
    facts = _facts(root)
    index_book = payload["index_book"]
    port_book = payload["portfolio_book"]
    results = payload["results"]
    ready = index_book.sum(axis=1) > 0.9
    start = ready[ready].index[0]
    view = results.loc[start:]
    index_view = index_book.loc[view.index]
    port_view = port_book.loc[view.index]
    order = index_view.iloc[-1].sort_values(ascending=False).index.tolist()
    book_growth = payload["results"].loc[view.index, "portfolio_equity"]
    sti_growth = payload["results"].loc[view.index, "index_equity"]
    book_growth = (book_growth / book_growth.iloc[0] * 100).round(2)
    sti_growth = (sti_growth / sti_growth.iloc[0] * 100).round(2)
    months = pd.Series(view.index, index=view.index).dt.to_period("M").dt.to_timestamp()
    keys = pd.Series(facts.index, index=facts.index).reindex(months.to_numpy()).ffill()
    flags = facts.reindex(months.to_numpy()).ffill()
    leaf = []
    month_keys = []
    for day, key, row in zip(view.index, keys.to_numpy(), flags.itertuples(index=False)):
        if pd.isna(key) or pd.isna(row.high_yield):
            leaf.append(-1)
            month_keys.append("")
            continue
        leaf.append(int(bool(row.high_yield)) * 2 + int(bool(row.wide_credit)))
        month_keys.append(pd.Timestamp(key).strftime("%Y-%m-%d"))
    fact_rows = {}
    for day, row in facts.iterrows():
        fact_rows[pd.Timestamp(day).strftime("%Y-%m-%d")] = {
            "y10": round(float(row["yield_10y"]), 2),
            "yMed": round(float(row["yield_median"]), 2),
            "high": bool(row["high_yield"]),
            "hy": round(float(row["hy"]), 2),
            "ig": round(float(row["ig"]), 2),
            "spread": round(float(row["credit"]), 2),
            "sMed": round(float(row["credit_median"]), 2),
            "wide": bool(row["wide_credit"]),
        }
    reviews = []
    for row in payload["reviews"].itertuples(index=False):
        reviews.append(
            {
                "date": pd.Timestamp(row.date).strftime("%Y-%m-%d"),
                "leaf": row.leaf,
                "mix": row.mix,
                "te": round(float(row.predicted_tracking_error), 6),
            }
        )
    window = results.loc[start:]
    index_total = float(window["index_equity"].iloc[-1] / window["index_equity"].iloc[0] - 1)
    port_total = float(window["portfolio_equity"].iloc[-1] / window["portfolio_equity"].iloc[0] - 1)
    excess = window["excess_return"].dropna()
    deviation = float(excess.std(ddof=0)) if len(excess) else float("nan")
    mixes, waiting = _mix_payload(payload["search"])
    return {
        "dates": [day.strftime("%Y-%m-%d") for day in view.index],
        "book": book_growth.tolist(),
        "sti": sti_growth.tolist(),
        "names": [NAMES.get(ticker, ticker) for ticker in order],
        "stiW": (index_view[order].fillna(0.0) * 10000).round().astype(int).values.tolist(),
        "bookW": (port_view[order].fillna(0.0) * 10000).round().astype(int).values.tolist(),
        "leaf": leaf,
        "month": month_keys,
        "facts": fact_rows,
        "years": _year_rows(window),
        "reviews": reviews,
        "mixes": mixes,
        "waiting": waiting,
        "summary": {
            "book": round(port_total, 6),
            "sti": round(index_total, 6),
            "excess": round(port_total - index_total, 6),
            "te": round(deviation * (252 ** 0.5), 6),
            "ir": round(float(excess.mean()) / deviation * (252 ** 0.5), 4) if deviation else None,
        },
    }


def main() -> None:
    data = _page_data(ROOT)
    template = (ROOT / "web" / "sti_template.html").read_text(encoding="utf-8")
    html = template.replace("/*@@DATA@@*/null", json.dumps(data, separators=(",", ":")))
    destination = ROOT / "web" / "public" / "sti.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    summary = data["summary"]
    print(f"Wrote {destination}")
    print(
        f"book {summary['book']:+.2%} proxy {summary['sti']:+.2%} "
        f"excess {summary['excess']:+.2%} te {summary['te']:.2%} ir {summary['ir']:.2f}"
    )


if __name__ == "__main__":
    main()
