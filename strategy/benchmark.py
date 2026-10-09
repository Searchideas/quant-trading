"""Benchmark download + performance metrics aligned with strategy analytics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ssl_utils import configure_ssl

configure_ssl()

import pandas as pd
import yfinance as yf

from analytics.performance import TRADING_DAYS, adjusted_sharpe, max_drawdown, month_lead, sharpe_ratio


class Benchmark:
    def __init__(
        self,
        symbol: str = "SPY",
        start_date: str | None = None,
        end_date: str | None = None,
        initial_capital: float = 100_000.0,
    ):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital

        self.prices = self.download_benchmark_data()
        self.returns = self.prices.pct_change().dropna()
        self.equity = initial_capital * (1 + self.returns).cumprod()
        self.stats = self.summarize()

    def download_benchmark_data(self) -> pd.Series:
        end = self.end_date or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
        data = yf.download(
            self.symbol,
            start=self.start_date,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if not data.empty:
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            close.name = self.symbol
            close.index.name = "Date"
            return close

        # Fallback when yfinance/curl SSL fails on Windows
        close = self._download_via_yahoo_chart(self.start_date, end)
        if close.empty:
            raise ValueError(f"No data for {self.symbol}")
        return close

    def _download_via_yahoo_chart(self, start: str | None, end: str) -> pd.Series:
        import requests

        start_ts = int(pd.Timestamp(start or "2020-01-01").timestamp())
        end_ts = int(pd.Timestamp(end).timestamp())
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{self.symbol}"
        params = {
            "period1": start_ts,
            "period2": end_ts,
            "interval": "1d",
            "events": "history",
        }
        resp = requests.get(
            url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=30
        )
        resp.raise_for_status()
        result = resp.json()["chart"]["result"]
        if not result:
            return pd.Series(dtype=float, name=self.symbol)
        payload = result[0]
        timestamps = payload.get("timestamp") or []
        closes = (payload.get("indicators") or {}).get("quote", [{}])[0].get("close") or []
        series = pd.Series(
            closes,
            index=pd.to_datetime(timestamps, unit="s"),
            name=self.symbol,
            dtype=float,
        ).dropna()
        idx = series.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        series.index = idx.normalize()
        series = series[~series.index.duplicated(keep="last")]
        series.index.name = "Date"
        return series

    def summarize(self) -> pd.Series:
        rets = self.returns
        equity = self.equity
        n = len(rets)
        years = n / TRADING_DAYS if n else float("nan")

        start = float(self.initial_capital)
        end = float(equity.iloc[-1]) if len(equity) else start
        total_return = end / start - 1
        cagr = (
            float((end / start) ** (1 / years) - 1) if years and years > 0 else float("nan")
        )
        vol = float(rets.std() * (TRADING_DAYS**0.5)) if len(rets) > 1 else float("nan")
        dd = max_drawdown(equity)
        max_dd = float(dd.min()) if len(dd) else float("nan")

        downside = rets[rets < 0]
        sortino = (
            float(rets.mean() / downside.std() * (TRADING_DAYS**0.5))
            if len(downside) > 1 and downside.std()
            else float("nan")
        )
        calmar = float(cagr / abs(max_dd)) if max_dd and max_dd != 0 else float("nan")

        return pd.Series(
            {
                "symbol": self.symbol,
                "initial_capital": start,
                "final_capital": end,
                "profit": end - start,
                "days": n,
                "years": round(years, 2) if years == years else years,
                "total_return": total_return,
                "cagr": cagr,
                "vol": vol,
                "sharpe": sharpe_ratio(rets),
                "adjusted_sharpe": adjusted_sharpe(rets),
                "sortino": sortino,
                "calmar": calmar,
                "max_drawdown": max_dd,
                "win_rate": float((rets > 0).mean()) if len(rets) else float("nan"),
            }
        )

    def compare(self, strategy_stats: pd.Series) -> pd.DataFrame:
        keys = [
            "total_return",
            "cagr",
            "vol",
            "sharpe",
            "adjusted_sharpe",
            "max_drawdown",
            "win_rate",
        ]
        rows = {}
        for k in keys:
            rows[k] = {
                "strategy": strategy_stats.get(k),
                "benchmark": self.stats.get(k),
            }
        table = pd.DataFrame(rows).T
        table["diff (strat - bm)"] = table["strategy"] - table["benchmark"]
        return table

    def print_comparison(
        self,
        strategy_stats: pd.Series,
        strategy_returns: pd.Series | None = None,
    ) -> pd.DataFrame:
        table = self.compare(strategy_stats)
        pct_keys = {"total_return", "cagr", "vol", "max_drawdown", "win_rate"}
        print(f"\n=== Strategy vs {self.symbol} ===")
        for metric, row in table.iterrows():
            if metric in pct_keys:
                print(
                    f"{metric:16s}: strategy {row['strategy']:+.2%} | "
                    f"{self.symbol} {row['benchmark']:+.2%} | "
                    f"diff {row['diff (strat - bm)']:+.2%}"
                )
            else:
                print(
                    f"{metric:16s}: strategy {row['strategy']:.2f} | "
                    f"{self.symbol} {row['benchmark']:.2f} | "
                    f"diff {row['diff (strat - bm)']:+.2f}"
                )
        if strategy_returns is not None:
            share, median = month_lead(strategy_returns, self.returns)
            table.loc["months_ahead"] = {
                "strategy": share,
                "benchmark": float("nan"),
                "diff (strat - bm)": float("nan"),
            }
            table.loc["median_month_excess"] = {
                "strategy": median,
                "benchmark": 0.0,
                "diff (strat - bm)": median,
            }
            print(f"{'months_ahead':16s}: {share:.1%}")
            print(f"{'median_month':16s}: {median:+.2%}")
        return table


if __name__ == "__main__":
    from config import get_market_config

    cfg = get_market_config()
    bm = Benchmark(
        cfg.benchmark_symbol,
        start_date=cfg.start_date,
        initial_capital=100_000.0,
    )
    print(f"=== {bm.symbol} Benchmark ===")
    for k, v in bm.stats.items():
        print(f"{k}: {v}")
