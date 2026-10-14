# Bloomberg Excel → SG raw data

Pull SGX historical data with Bloomberg formulas into the CSV layout this project expects.

## What you get

| File | Role |
|------|------|
| `SG_Bloomberg_Historical_Pull.xlsx` | Bloomberg Excel template (BDH / BDP / BDS) |
| `SG_Bloomberg_Historical_Pull_5y.xlsx` | Same template with rolling **5-year** `StartDate` |
| `create_bbg_template.py` | Rebuilds the workbook from `config.SG_TICKERS` |
| `bbg_to_csv.py` | Converts refreshed Excel → `data/sg/raw/*.csv` |

## Project raw inputs (SG)

| CSV | Content | Bloomberg field |
|-----|---------|-----------------|
| `price_data.csv` | Daily adjusted close | `PX_LAST` + cash/cap adj overrides |
| `book_data.csv` | Monthly BVPS | `BOOK_VAL_PER_SH` (`Per=M` — not `BQ`) |
| `categorizations_data.csv` | Sector + mkt cap | `GICS_SECTOR_NAME`, `CUR_MKT_CAP` |
| `dividend_data.csv` | Cash DPS on ex-date | `BDS` + `DVD_HIST_ALL` (dividend history) |

Tickers stay as Yahoo-style `.SI` (e.g. `D05.SI`). Bloomberg codes are `D05 SP Equity`.

## Setup

1. On a **Bloomberg Terminal** PC, open `SG_Bloomberg_Historical_Pull.xlsx` in Excel.
2. Check **Control** sheet: `StartDate` defaults to `=EDATE(TODAY(),-60)` (rolling **5 years**); `EndDate` is `=TODAY()-1`.
3. Bloomberg menu → **Refresh Workbook**.
4. Wait until Prices / Book / Categories show numbers (not `#N/A Requesting Data...`).
5. For dividends: on **Dividends**, set `SelectYahoo`, refresh `BDS`, paste Ex-Date + Amount into the long table (`Date`, `Ticker`, `DPS`).
6. Save the workbook.

## Export into the pipeline

```powershell
cd "quant-trading"
python bloomberg/bbg_to_csv.py --xlsx bloomberg/SG_Bloomberg_Historical_Pull.xlsx
python main.py --market sg --skip-download
```

## Rebuild template after ticker list changes

```powershell
python bloomberg/create_bbg_template.py
```

## Formula cheat-sheet

**Adjusted price (one ticker, dates + values):**
```excel
=BDH("D05 SP Equity","PX_LAST",$B$2,$B$3,"Dir=V","Dts=S","Sort=A","CshAdjNormal=Y","CshAdjAbnormal=Y","CapChg=Y","Fill=P","Days=A")
```

**Price values only (align to same dates):**
```excel
=BDH("O39 SP Equity","PX_LAST",$B$2,$B$3,"Dir=V","Dts=H","Sort=A","CshAdjNormal=Y","CshAdjAbnormal=Y","CapChg=Y","Fill=P","Days=A")
```

**Quarterly book:**
```excel
=BDH("D05 SP Equity","BOOK_VAL_PER_SH",$B$2,$B$3,"Dir=V","Dts=S","Sort=A","Per=M","Fill=P")
```

**Static sector / cap:**
```excel
=BDP("D05 SP Equity","GICS_SECTOR_NAME")
=BDP("D05 SP Equity","CUR_MKT_CAP")
```

**Dividend history:**
```excel
=BDS("D05 SP Equity","DVD_HIST_ALL","Headers=Y","DVD_START_DT=20210901","DVD_END_DT=20260904")
```

Do **not** use `StartDateTime` / `EndDateTime` with `DVD_HIST_ALL` — Bloomberg returns `Invalid override field: EndDateTime`.

## Notes

- Needs Bloomberg Excel Add-in; formulas will not calculate on a normal PC.
- If `GICS_SECTOR_NAME` is blank, try `INDUSTRY_SECTOR`.
- REITs: if `BOOK_VAL_PER_SH` is empty, check NAV fields with your market-data contact.
- This project does **not** need OHLCV — adjusted close only.
