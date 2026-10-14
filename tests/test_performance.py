import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest
from config import INITIAL_CAPITAL, get_paths
from analytics.performance import market_regression, max_drawdown, summarize


def _load_backtest(market: str | None = None) -> pd.DataFrame:
    _, processed_dir = get_paths(market)
    path = processed_dir / "backtest_results.csv"
    if not path.exists():
        pytest.skip(f"Run main.py first — missing {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def test_max_drawdown_on_real_equity():
    bt = _load_backtest()
    dd = max_drawdown(bt["equity"])

    assert dd.min() <= 0.0
    assert dd.min() >= -1.0
    assert dd.iloc[0] == 0.0


def test_summarize_on_real_backtest():
    bt = _load_backtest()
    summary = summarize(bt, initial_capital=INITIAL_CAPITAL)

    assert summary["initial_capital"] == INITIAL_CAPITAL
    assert summary["final_capital"] == pytest.approx(float(bt["equity"].iloc[-1]))
    assert summary["max_drawdown"] == pytest.approx(float(max_drawdown(bt["equity"]).min()))
    assert summary["days"] == len(bt["return"].dropna())
    assert 0.0 <= summary["win_rate"] <= 1.0


def test_real_equity_starts_at_initial_capital():
    bt = _load_backtest()
    assert bt["equity"].iloc[0] == pytest.approx(INITIAL_CAPITAL)


def test_market_regression_recovers_a_known_line():
    index = pd.bdate_range("2020-01-01", periods=40)
    market = pd.Series(np.linspace(-0.02, 0.02, 40), index=index)
    strategy = 0.0004 + 1.2 * market
    fit = market_regression(strategy, market)

    assert fit["days"] == 40
    assert fit["beta"] == pytest.approx(1.2)
    assert fit["alpha"] == pytest.approx(0.0004)
    assert fit["alpha_se"] == pytest.approx(0.0, abs=1e-12)
    assert fit["residual_vol"] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(fit["information_ratio"])


def test_market_regression_matches_numpy_and_scales_alpha():
    index = pd.bdate_range("2020-01-01", periods=60)
    market = pd.Series(np.sin(np.linspace(0, 8, 60)) * 0.01, index=index)
    noise = pd.Series(np.cos(np.linspace(0, 5, 60)) * 0.002, index=index)
    strategy = 0.0003 + 0.8 * market + noise
    fit = market_regression(strategy, market)

    design = np.column_stack([np.ones(60), market.to_numpy()])
    alpha, beta = np.linalg.lstsq(design, strategy.to_numpy(), rcond=None)[0]
    assert fit["alpha"] == pytest.approx(alpha)
    assert fit["beta"] == pytest.approx(beta)
    assert fit["alpha_annualized"] == pytest.approx(alpha * 252)
    assert fit["alpha_annualized_se"] == pytest.approx(fit["alpha_se"] * 252)
    assert fit["alpha_low"] < fit["alpha_annualized"] < fit["alpha_high"]
    assert fit["information_ratio"] == pytest.approx(
        fit["alpha_annualized"] / fit["residual_vol"]
    )


def test_page_market_table_names_beta_alpha_and_information_ratio():
    from web.build_page import _model_cells, market_table

    index = pd.bdate_range("2024-01-01", periods=80)
    market = pd.Series(np.sin(np.linspace(0, 6, 80)) * 0.01, index=index)
    strategy = 0.0002 + 1.1 * market
    rows = {row["metric"]: row["value"] for row in market_table(strategy, market)}
    cells = _model_cells(strategy, market)

    assert rows["Beta"].startswith("1.10")
    assert "se 0.0%" in rows["Alpha per year"]
    assert "to" in rows["Alpha, 95% range"]
    assert "Unexplained volatility" in rows
    assert "Information ratio" in rows
    assert cells["beta"] == "1.10"
    assert cells["alpha"].startswith("+")


def test_equal_series_have_beta_one_and_zero_information_ratio():
    index = pd.bdate_range("2020-01-01", periods=30)
    market = pd.Series(np.linspace(0.001, 0.02, 30), index=index)
    fit = market_regression(market, market)

    assert fit["beta"] == pytest.approx(1.0)
    assert fit["alpha"] == pytest.approx(0.0, abs=1e-12)
    assert fit["information_ratio"] == pytest.approx(0.0, abs=1e-8)