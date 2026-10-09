import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1


def sharpe_ratio(rets: pd.Series, periods: int = TRADING_DAYS) -> float:
    if rets.std() == 0 or len(rets) < 2:
        return np.nan
    return float(rets.mean() / rets.std() * np.sqrt(periods))


def compounded_monthly(rets: pd.Series) -> pd.Series:
    """Month return from compounding the daily returns inside each month."""
    clean = rets.dropna()
    if clean.empty:
        return clean
    return (1 + clean).resample("ME").prod() - 1


def month_lead(strategy: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
    """Share of months the strategy beat the benchmark, and the median monthly gap."""
    monthly = pd.concat(
        [compounded_monthly(strategy), compounded_monthly(benchmark)],
        axis=1,
        keys=["strategy", "benchmark"],
    ).dropna()
    if monthly.empty:
        return float("nan"), float("nan")
    excess = monthly["strategy"] - monthly["benchmark"]
    share = float((monthly["strategy"] > monthly["benchmark"]).mean())
    return share, float(excess.median())


def adjusted_sharpe(rets: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Lo (2002) AR(1) adjustment for autocorrelated returns."""
    sr = sharpe_ratio(rets, periods)
    if np.isnan(sr):
        return np.nan
    rho1 = rets.autocorr(lag=1)
    if pd.isna(rho1):
        rho1 = 0.0
    return float(sr * np.sqrt((1 - rho1) / (1 + rho1)))


def summarize(bt: pd.DataFrame, initial_capital: float = 100_000.0) -> pd.Series:
    """Stats from BacktestEngine.run() output."""
    rets = bt["return"].dropna()
    equity = bt["equity"].dropna()
    n = len(rets)
    years = n / TRADING_DAYS

    start = float(initial_capital)
    end = float(equity.iloc[-1])
    total_return = end / start - 1
    cagr = float((end / start) ** (1 / years) - 1) if years > 0 else np.nan
    vol = float(rets.std() * np.sqrt(TRADING_DAYS))
    sharpe = sharpe_ratio(rets)
    adj_sharpe = adjusted_sharpe(rets)
    dd = max_drawdown(equity)
    win_rate = float((rets > 0).mean())

    return pd.Series(
        {
            "initial_capital": start,
            "final_capital": end,
            "profit": end - start,
            "days": n,
            "years": round(years, 2),
            "total_return": total_return,
            "cagr": cagr,
            "vol": vol,
            "sharpe": sharpe,
            "adjusted_sharpe": adj_sharpe,
            "max_drawdown": float(dd.min()),
            "win_rate": win_rate,
            "trade_days": int((bt["turnover"] > 0).sum()),
            "hold_days": int((bt["turnover"] == 0).sum()),
            "total_cost_sgd": float(bt["cost_sgd"].sum()) if "cost_sgd" in bt.columns else 0.0,
        }
    )
