import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest
from config import INITIAL_CAPITAL, get_paths
from analytics.performance import max_drawdown, summarize


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