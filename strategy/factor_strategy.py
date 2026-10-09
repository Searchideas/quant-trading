from pathlib import Path

import numpy as np
import pandas as pd

from paths import get_paths


def _read_sector(processed_dir: Path, name: str) -> pd.DataFrame | None:
    path = processed_dir / f"sector_{name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_strategy_data(market: str | None = None) -> dict[str, pd.DataFrame | pd.Series]:
    raw_dir, processed_dir = get_paths(market)
    categorizations = pd.read_csv(raw_dir / "categorizations_data.csv", index_col=0)
    sector_market_cap = (
        categorizations.groupby("sector")["market_cap"].sum().sort_values(ascending=False)
    )

    return {
        "price_data": pd.read_csv(raw_dir / "price_data.csv", index_col=0, parse_dates=True),
        "dividend_data": pd.read_csv(raw_dir / "dividend_data.csv", index_col=0, parse_dates=True),
        "categorizations": categorizations,
        "factors_combined": pd.read_csv(
            processed_dir / "factors_combined.csv", index_col=0, parse_dates=True
        ),
        "sector_value": pd.read_csv(
            processed_dir / "sector_value.csv", index_col=0, parse_dates=True
        ),
        "sector_momentum": pd.read_csv(
            processed_dir / "sector_momentum.csv", index_col=0, parse_dates=True
        ),
        "sector_volatility": pd.read_csv(
            processed_dir / "sector_volatility.csv", index_col=0, parse_dates=True
        ),
        "sector_market_cap": sector_market_cap,
        "sector_dividend": pd.read_csv(
            processed_dir / "sector_dividend.csv", index_col=0, parse_dates=True
        ),
        "sector_growth": _read_sector(processed_dir, "growth"),
        "sector_quality": _read_sector(processed_dir, "quality"),
    }


def normalize_factors(
    factors_combined,
    categorizations,
    sector_value,
    sector_dividend,    # 4th
    sector_momentum,    # 5th
    sector_volatility,  # 6th
    sector_growth=None,
    sector_quality=None,
) -> tuple[pd.DataFrame, list[str]]:
    """Z-score each factor vs its sector mean/std."""
    sector_dfs = {
        "value": sector_value,
        "momentum": sector_momentum,
        "volatility": sector_volatility,
        "dividend": sector_dividend,
        "growth": sector_growth,
        "quality": sector_quality,
    }

    normalized_cols: dict[str, pd.Series] = {}
    tickers: set[str] = set()

    for col in factors_combined.columns:
        ticker, factor = col.rsplit("_", 1)
        tickers.add(ticker)
        sector = categorizations.loc[ticker, "sector"]
        sector_df = sector_dfs.get(factor)
        if sector_df is None:
            normalized_cols[col] = pd.Series(0.0, index=factors_combined.index)
            continue
        mean_col = f"{sector}_mean"
        std_col = f"{sector}_std"
        normalized_cols[col] = (factors_combined[col] - sector_df[mean_col]) / sector_df[std_col]

    return pd.DataFrame(normalized_cols, index=factors_combined.index), sorted(tickers)


def _factor_zscore(normalized: pd.DataFrame, ticker: str, factor: str) -> pd.Series:
    """Return normalized z-score column, or zeros if factor missing for ticker."""
    col = f"{ticker}_{factor}"
    if col in normalized.columns:
        return normalized[col].fillna(0)
    return pd.Series(0.0, index=normalized.index)


def combine_final_score(
    normalized: pd.DataFrame,
    tickers: list[str],
    momentum_weightage: float,
    value_weightage: float,
    volatility_weightage: float,
    dividend_weightage: float,
    growth_weightage: float = 0.0,
    quality_weightage: float = 0.0,
) -> pd.DataFrame:
    """Blend normalized factor z-scores into one score per ticker."""
    final_score = pd.DataFrame(index=normalized.index)
    for ticker in tickers:
        final_score[ticker] = (
            momentum_weightage * _factor_zscore(normalized, ticker, "momentum")
            + value_weightage * _factor_zscore(normalized, ticker, "value")
            + volatility_weightage * _factor_zscore(normalized, ticker, "volatility")
            + dividend_weightage * _factor_zscore(normalized, ticker, "dividend")
            + growth_weightage * _factor_zscore(normalized, ticker, "growth")
            + quality_weightage * _factor_zscore(normalized, ticker, "quality")
        )
    return final_score.fillna(0)


def factor_strategy(
    factors_combined: pd.DataFrame,
    categorizations: pd.DataFrame,
    sector_value: pd.DataFrame,
    sector_momentum: pd.DataFrame,
    sector_volatility: pd.DataFrame,
    sector_dividend: pd.DataFrame,
    sector_growth: pd.DataFrame | None = None,
    sector_quality: pd.DataFrame | None = None,
    momentum_weightage: float = 0.4,
    value_weightage: float = 0.3,
    volatility_weightage: float = 0.3,
    dividend_weightage: float = 0.0,
    growth_weightage: float = 0.0,
    quality_weightage: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Z-score normalize factors by sector, then combine into a weighted final score."""
    normalized, tickers = normalize_factors(
        factors_combined,
        categorizations,
        sector_value,
        sector_dividend,
        sector_momentum,
        sector_volatility,
        sector_growth,
        sector_quality,
    )
    final_score = combine_final_score(
        normalized,
        tickers,
        momentum_weightage,
        value_weightage,
        volatility_weightage,
        dividend_weightage,
        growth_weightage,
        quality_weightage,
    )
    return normalized, final_score


def _cap_sector_weights(raw: pd.Series, max_weight: float = 0.35) -> pd.Series:
    """Cap sector weights and redistribute excess proportionally."""
    weights = raw.astype(float).copy()
    while (weights > max_weight + 1e-12).any():
        over = weights > max_weight
        excess = (weights[over] - max_weight).sum()
        weights[over] = max_weight
        under = ~over
        if not under.any() or excess <= 0:
            break
        weights[under] += excess * (weights[under] / weights[under].sum())
    return weights


def select_sector_best_weights(
    final_score: pd.DataFrame,
    categorizations: pd.DataFrame,
    sector_market_cap: pd.Series,
    max_sector_weight: float = 1.0,
    top_n_per_sector: int = 1,
) -> pd.DataFrame:
    """Pick top N tickers per sector by score; allocate by sector market cap (capped).

    - Sector bucket size = sector's share of total market cap (max 35% per sector)
    - Within a sector (if top_n > 1), split bucket by each stock's market cap
    - Rebalance only when the selected ticker set changes
    """
    if top_n_per_sector < 1:
        raise ValueError("top_n_per_sector must be at least 1")

    ticker_to_sector = categorizations["sector"]
    stock_market_cap = categorizations["market_cap"]
    columns = list(final_score.columns)
    position = {ticker: i for i, ticker in enumerate(columns)}
    sector_members = []
    for sector in sector_market_cap[sector_market_cap > 0].index:
        members = [
            position[ticker]
            for ticker in ticker_to_sector[ticker_to_sector == sector].index
            if ticker in position
        ]
        if members:
            sector_members.append((sector, np.asarray(members, dtype=int)))

    stock_caps = np.array(
        [
            float(stock_market_cap[ticker]) if ticker in stock_market_cap.index else np.nan
            for ticker in columns
        ],
        dtype=float,
    )
    scores = final_score.to_numpy(dtype=float)
    out = np.zeros(scores.shape, dtype=float)
    current = np.zeros(scores.shape[1], dtype=float)
    previous: tuple[int, ...] | None = None

    for i, row in enumerate(scores):
        chosen: list[tuple[str, np.ndarray]] = []
        chosen_ids: list[int] = []
        for sector, members in sector_members:
            values = row[members]
            valid = np.flatnonzero(~np.isnan(values))
            if valid.size == 0:
                continue
            order = np.argsort(-values[valid], kind="stable")
            picked = members[valid[order[:top_n_per_sector]]]
            chosen.append((sector, picked))
            chosen_ids.extend(picked.tolist())

        selection = tuple(sorted(chosen_ids))
        if selection and selection != previous:
            previous = selection
            current = np.zeros(scores.shape[1], dtype=float)
            sector_names = [sector for sector, _ in chosen]
            sector_w = _cap_sector_weights(
                sector_market_cap.loc[sector_names] / sector_market_cap.loc[sector_names].sum(),
                max_sector_weight,
            )
            for sector, picked in chosen:
                caps = np.nan_to_num(stock_caps[picked], nan=0.0)
                if caps.sum() <= 0:
                    caps = np.ones(len(picked))
                current[picked] = float(sector_w.loc[sector]) * (caps / caps.sum())
        out[i] = current

    return pd.DataFrame(out, index=final_score.index, columns=final_score.columns)
