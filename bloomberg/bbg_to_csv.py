"""Convert Bloomberg Excel template (values) into project raw CSVs.

Usage (from quant-trading/):
  python bloomberg/bbg_to_csv.py --xlsx bloomberg/SG_Bloomberg_Historical_Pull.xlsx

Reads sheets: Prices, Book, Categories, and optional Dividends long table.
Writes: data/sg/raw/price_data.csv, book_data.csv, categorizations_data.csv,
        dividend_data.csv (if long table present).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from config import get_paths


def _read_sheet(xlsx: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name=sheet, engine="openpyxl")
    return df


def _clean_wide(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Normalize a wide Date × ticker matrix for the pipeline."""
    if date_col not in df.columns:
        # Sometimes first column is unnamed after spill
        df = df.rename(columns={df.columns[0]: date_col})
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col).sort_index()
    # Drop non-ticker helper columns
    bad = [c for c in df.columns if str(c).startswith("Unnamed")]
    df = df.drop(columns=bad, errors="ignore")
    # Coerce numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df.index.name = "Date"
    return df


def _merge_date_value_blocks(
    xlsx: Path,
    sheet: str,
    cols_per: int = 2,
    label_row_hint: int = 2,
) -> pd.DataFrame:
    """Parse side-by-side Date|Value BDH blocks → wide Date × ticker matrix."""
    raw = pd.read_excel(xlsx, sheet_name=sheet, header=None, engine="openpyxl")
    # Find label row with .SI tickers
    label_row = None
    for i in range(min(6, len(raw))):
        vals = [str(x) for x in raw.iloc[i].tolist() if pd.notna(x)]
        if any(".SI" in v for v in vals):
            label_row = i
            break
    if label_row is None:
        raise ValueError(f"{sheet}: no Yahoo .SI labels found")

    series_list: list[pd.Series] = []
    ncols = raw.shape[1]
    col = 0
    while col < ncols:
        label = raw.iloc[label_row, col]
        if not (isinstance(label, str) and ".SI" in label):
            col += 1
            continue
        ticker = label.strip()
        # Data starts at label_row+1 (BDH spill: Date | Value)
        block = raw.iloc[label_row + 1 :, [col, col + 1]].copy()
        block.columns = ["Date", ticker]
        block["Date"] = pd.to_datetime(block["Date"], errors="coerce")
        block[ticker] = pd.to_numeric(block[ticker], errors="coerce")
        block = block.dropna(subset=["Date"])
        # Drop trailing empty values but keep dates with valid numbers when present
        block = block.dropna(subset=[ticker], how="all")
        if not block.empty:
            s = block.set_index("Date")[ticker]
            s = s[~s.index.duplicated(keep="last")]
            series_list.append(s)
        col += cols_per

    if not series_list:
        raise ValueError(f"{sheet}: no BDH blocks parsed — refresh Bloomberg first?")

    wide = pd.concat(series_list, axis=1, join="outer").sort_index()
    wide.index.name = "Date"
    return wide


def export_prices(xlsx: Path, out: Path) -> Path:
    try:
        df = _merge_date_value_blocks(xlsx, "Prices", cols_per=2)
    except Exception as e:
        print(f"Block parse failed ({e}); trying legacy wide Prices…")
        df = _clean_wide(_read_sheet(xlsx, "Prices"))
    path = out / "price_data.csv"
    df.to_csv(path)
    print(f"Wrote {path}  shape={df.shape}")
    return path


def export_book(xlsx: Path, out: Path) -> Path:
    try:
        df = _merge_date_value_blocks(xlsx, "Book", cols_per=2)
    except Exception as e:
        print(f"Block parse failed ({e}); trying legacy wide Book…")
        df = _clean_wide(_read_sheet(xlsx, "Book"))
    path = out / "book_data.csv"
    df.to_csv(path)
    print(f"Wrote {path}  shape={df.shape}")
    return path


def export_categories(xlsx: Path, out: Path) -> Path:
    df = _read_sheet(xlsx, "Categories")
    # Expect Ticker, sector, market_cap
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for want in ("ticker", "sector", "market_cap"):
        if want in cols:
            rename[cols[want]] = "Ticker" if want == "ticker" else want
        elif want.title() in df.columns or want in df.columns:
            pass
    df = df.rename(columns={c: ("Ticker" if c.lower() == "ticker" else c) for c in df.columns})
    # Standardize names
    mapping = {}
    for c in df.columns:
        cl = c.lower().replace(" ", "_")
        if cl == "ticker":
            mapping[c] = "Ticker"
        elif cl == "sector":
            mapping[c] = "sector"
        elif cl in ("market_cap", "mkt_cap", "cur_mkt_cap"):
            mapping[c] = "market_cap"
    df = df.rename(columns=mapping)
    need = ["Ticker", "sector", "market_cap"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Categories sheet missing columns {missing}; got {list(df.columns)}")
    df = df[need].dropna(subset=["Ticker"])
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df = df.set_index("Ticker")
    path = out / "categorizations_data.csv"
    df.to_csv(path)
    print(f"Wrote {path}  shape={df.shape}")
    return path


def export_dividends(xlsx: Path, out: Path) -> Path | None:
    """Parse Dividends sheet BDS DVD_HIST_ALL blocks → wide dividend_data.csv."""
    path = _export_dividends_hist_blocks(xlsx, out)
    if path is not None:
        return path
    return export_dividends_long(xlsx, out)


def _export_dividends_hist_blocks(xlsx: Path, out: Path) -> Path | None:
    """Read side-by-side DVD_HIST_ALL blocks (row3=Yahoo ticker, row4+=BDS spill)."""
    try:
        raw = pd.read_excel(xlsx, sheet_name="Dividends", header=None, engine="openpyxl")
    except ValueError:
        return None

    if raw.empty:
        return None

    # Label row: Excel row 3 → pandas index 2
    label_row = None
    for i in range(min(6, len(raw))):
        vals = [str(x) for x in raw.iloc[i].tolist() if pd.notna(x)]
        if any(".SI" in v for v in vals):
            label_row = i
            break
    if label_row is None:
        return None

    records: list[dict] = []
    ncols = raw.shape[1]
    col = 0
    while col < ncols:
        label = raw.iloc[label_row, col]
        if not (isinstance(label, str) and ".SI" in label):
            col += 1
            continue

        ticker = label.strip()
        # Header row is usually the next non-empty row in this block (BDS Headers=Y)
        header_row = None
        date_j = amt_j = type_j = None
        for r in range(label_row + 1, min(label_row + 6, len(raw))):
            headers = [str(x).strip().lower() if pd.notna(x) else "" for x in raw.iloc[r, col : col + 10]]
            # Find ex-date / amount / type within this block
            for j, h in enumerate(headers):
                if h in ("ex-date", "ex date", "ex_date", "exdate") or (
                    "ex" in h and "date" in h
                ):
                    date_j = j
                if h in ("dividend amount", "div amount", "amount", "dps", "dividend"):
                    if "type" not in h and "date" not in h:
                        amt_j = j
                if h in ("dividend type", "div type", "type"):
                    type_j = j
            if date_j is not None and amt_j is not None:
                header_row = r
                break

        if header_row is None:
            col += 8
            continue

        for r in range(header_row + 1, len(raw)):
            date_val = raw.iloc[r, col + date_j]
            amt_val = raw.iloc[r, col + amt_j]
            if pd.isna(date_val) and pd.isna(amt_val):
                # stop at first fully empty data row in block (end of spill)
                # but allow trailing blanks mid-series — break only if next few empty
                if r + 1 < len(raw) and pd.isna(raw.iloc[r + 1, col + date_j]):
                    break
                continue
            typ = ""
            if type_j is not None:
                tval = raw.iloc[r, col + type_j]
                typ = str(tval) if pd.notna(tval) else ""
            if "stock" in typ.lower():
                continue
            dt = pd.to_datetime(date_val, errors="coerce")
            dps = pd.to_numeric(amt_val, errors="coerce")
            if pd.isna(dt) or pd.isna(dps):
                continue
            records.append({"Date": dt, "Ticker": ticker, "DPS": float(dps), "Type": typ})

        col += 8

    if not records:
        print("Dividends DVD_HIST blocks found but no parsed rows (refresh BDS first?)")
        return None

    body = pd.DataFrame(records)
    wide = body.pivot_table(index="Date", columns="Ticker", values="DPS", aggfunc="sum")
    wide = wide.sort_index()
    wide.index.name = "Date"
    path = out / "dividend_data.csv"
    wide.to_csv(path)
    print(f"Wrote {path}  shape={wide.shape} (DVD_HIST_ALL blocks, {len(body)} events)")
    return path


def export_dividends_long(xlsx: Path, out: Path) -> Path | None:
    """Pivot Dividends_Detail / legacy long table (Date, Ticker, DPS) → wide dividend_data.csv."""
    for sheet in ("Dividends", "Dividends_Detail"):
        try:
            raw = pd.read_excel(
                xlsx,
                sheet_name=sheet,
                header=None,
                engine="openpyxl",
            )
        except ValueError:
            continue

        header_row = None
        for i in range(min(40, len(raw))):
            row = [str(x).strip().lower() for x in raw.iloc[i].tolist()]
            if "date" in row and "ticker" in row and ("dps" in row or "dividend" in row):
                header_row = i
                break
        if header_row is None:
            continue

        headers = [str(x).strip() for x in raw.iloc[header_row].tolist()]

        def find_col(names: tuple[str, ...]) -> int | None:
            for j, h in enumerate(headers):
                if h.lower() in names:
                    return j
            return None

        date_j = find_col(("date",))
        tick_j = find_col(("ticker", "yahoo ticker"))
        dps_j = find_col(("dps", "dividend", "dividend amount", "amount"))
        type_j = find_col(("type", "dividend type", "dvd type"))
        if date_j is None or tick_j is None or dps_j is None:
            continue

        cols = [date_j, tick_j, dps_j] + ([type_j] if type_j is not None else [])
        body = raw.iloc[header_row + 1 :, cols].copy()
        if type_j is not None:
            body.columns = ["Date", "Ticker", "DPS", "Type"]
            body = body[~body["Type"].astype(str).str.contains("stock", case=False, na=False)]
            body = body[["Date", "Ticker", "DPS"]]
        else:
            body.columns = ["Date", "Ticker", "DPS"]

        body["Date"] = pd.to_datetime(body["Date"], errors="coerce")
        body["DPS"] = pd.to_numeric(body["DPS"], errors="coerce")
        body = body.dropna(subset=["Date", "Ticker", "DPS"])
        if body.empty:
            continue

        wide = body.pivot_table(index="Date", columns="Ticker", values="DPS", aggfunc="sum")
        wide = wide.sort_index()
        wide.index.name = "Date"
        path = out / "dividend_data.csv"
        wide.to_csv(path)
        print(f"Wrote {path}  shape={wide.shape} (long table from {sheet})")
        return path

    print("No usable Dividends data — skip dividend_data.csv")
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Bloomberg Excel → SG raw CSVs")
    p.add_argument(
        "--xlsx",
        type=Path,
        default=Path(__file__).with_name("SG_Bloomberg_Historical_Pull.xlsx"),
        help="Path to refreshed Bloomberg workbook",
    )
    p.add_argument("--market", default="sg")
    args = p.parse_args()

    xlsx = args.xlsx.resolve()
    if not xlsx.exists():
        raise SystemExit(f"Workbook not found: {xlsx}")

    raw_dir, _ = get_paths(args.market)
    raw_dir.mkdir(parents=True, exist_ok=True)

    export_prices(xlsx, raw_dir)
    export_book(xlsx, raw_dir)
    export_categories(xlsx, raw_dir)
    export_dividends(xlsx, raw_dir)
    print(f"Done. Raw dir: {raw_dir}")
    print("Next: python main.py --market sg --skip-download")


if __name__ == "__main__":
    main()
