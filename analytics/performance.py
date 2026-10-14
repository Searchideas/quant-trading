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


def _daily(series: pd.Series) -> pd.Series:
    out = series.copy()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize()
    return out[~out.index.duplicated(keep="last")]


def market_regression(strategy: pd.Series, market: pd.Series) -> pd.Series:
    """Fit strategy = alpha + beta * market + unexplained.

    Alpha is the daily intercept. Its annual figure is that intercept times 252,
    and the standard error scales the same way. The 95% range uses 1.96 standard
    errors. The information ratio is annualized alpha divided by the annualized
    volatility of the unexplained return.
    """
    names = [
        "days",
        "alpha",
        "alpha_se",
        "alpha_t",
        "alpha_annualized",
        "alpha_annualized_se",
        "alpha_low",
        "alpha_high",
        "beta",
        "beta_se",
        "residual_vol",
        "information_ratio",
    ]
    blank = pd.Series({name: np.nan for name in names})
    paired = pd.concat(
        [_daily(strategy), _daily(market)],
        axis=1,
        keys=["strategy", "market"],
    ).dropna()
    y = paired["strategy"].to_numpy(dtype=float)
    x = paired["market"].to_numpy(dtype=float)
    n = len(y)
    if n < 3 or float(np.dot(x - x.mean(), x - x.mean())) == 0.0:
        blank["days"] = n
        return blank

    design = np.column_stack([np.ones(n), x])
    xtx = design.T @ design
    try:
        alpha, beta = np.linalg.solve(xtx, design.T @ y)
    except np.linalg.LinAlgError:
        blank["days"] = n
        return blank
    residual = y - (alpha + beta * x)
    sse = float(np.dot(residual, residual))
    if sse <= 1e-18:
        alpha_se = 0.0
        beta_se = 0.0
        residual_daily = 0.0
    else:
        sigma2 = sse / (n - 2)
        covariance = sigma2 * np.linalg.inv(xtx)
        alpha_se = float(np.sqrt(max(covariance[0, 0], 0.0)))
        beta_se = float(np.sqrt(max(covariance[1, 1], 0.0)))
        residual_daily = float(np.sqrt(sigma2))
    annualized = float(alpha) * TRADING_DAYS
    annualized_se = alpha_se * TRADING_DAYS
    if residual_daily == 0.0:
        ratio = 0.0 if abs(float(alpha)) <= 1e-10 else np.nan
    else:
        ratio = float(alpha) / residual_daily * np.sqrt(TRADING_DAYS)
    return pd.Series(
        {
            "days": n,
            "alpha": float(alpha),
            "alpha_se": alpha_se,
            "alpha_t": float(alpha) / alpha_se if alpha_se else np.nan,
            "alpha_annualized": annualized,
            "alpha_annualized_se": annualized_se,
            "alpha_low": annualized - 1.96 * annualized_se,
            "alpha_high": annualized + 1.96 * annualized_se,
            "beta": float(beta),
            "beta_se": beta_se,
            "residual_vol": residual_daily * np.sqrt(TRADING_DAYS),
            "information_ratio": ratio,
        }
    )


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
