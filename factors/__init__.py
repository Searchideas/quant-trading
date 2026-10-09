from pathlib import Path

import pandas as pd

from .fundamentals import growth_and_quality, load_statement_frames
from .io import load_book_data, load_categorizations, load_price_data, load_dividend_data
from .momentum import MomentumFactor
from .value import ValueFactor
from .volatility import VolatilityFactor
from .dividend import DividendFactor

__all__ = [
    "ValueFactor",
    "MomentumFactor",
    "VolatilityFactor",
    "DividendFactor",
    "load_price_data",
    "load_book_data",
    "load_categorizations",
    "factor_processing",
    "sector_mean",
]


def sector_mean(
    factor_df: pd.DataFrame, categorizations: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-sectional mean and std of a factor within each sector over time."""
    ticker_to_sector = categorizations["sector"]
    common = factor_df.columns.intersection(ticker_to_sector.index)
    grouped = factor_df[common].T.groupby(ticker_to_sector[common])
    return grouped.mean().T, grouped.std().T


def factor_processing(
    price_data: pd.DataFrame,
    book_data: pd.DataFrame,
    dividend_data: pd.DataFrame,
    market: str | None = None,
    processed_dir: Path | None = None,
    categorizations: pd.DataFrame | None = None,
) -> None:
    """Compute all factors and save to data/{market}/processed/."""
    from paths import ensure_market_dirs

    if processed_dir is None:
        _, processed_dir = ensure_market_dirs(market)
    else:
        processed_dir = Path(processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)

    if categorizations is None:
        categorizations = load_categorizations(market)

    value_df = ValueFactor(price_data, book_data).calculate_value_factor()
    momentum_df = MomentumFactor(price_data).calculate_momentum()
    volatility_df = VolatilityFactor(price_data).calculate_volatility()
    dividend_df = DividendFactor(price_data, dividend_data).calculate_dividend_factor()
    growth_df = pd.DataFrame(index=price_data.index)
    quality_df = pd.DataFrame(index=price_data.index)
    statements = load_statement_frames(market)
    if statements is None:
        print("Earnings statements not found. Growth and quality were skipped.")
    else:
        growth_df, quality_df = growth_and_quality(
            statements["quarterly_income"],
            statements["quarterly_shares"],
            statements["quarterly_equity"],
            statements["annual_income"],
            statements["annual_shares"],
            statements["annual_equity"],
            price_data.index,
        )
    value_df.to_csv(processed_dir / "value_data.csv")
    momentum_df.to_csv(processed_dir / "momentum_data.csv")
    volatility_df.to_csv(processed_dir / "volatility_data.csv")
    dividend_df.to_csv(processed_dir / "dividend_data.csv")
    growth_df.to_csv(processed_dir / "growth_data.csv")
    quality_df.to_csv(processed_dir / "quality_data.csv")
    factor_frames = [
        dividend_df.add_suffix("_dividend"),
        value_df.add_suffix("_value"),
        momentum_df.add_suffix("_momentum"),
        volatility_df.add_suffix("_volatility"),
    ]
    if not growth_df.empty:
        factor_frames.append(growth_df.add_suffix("_growth"))
    if not quality_df.empty:
        factor_frames.append(quality_df.add_suffix("_quality"))
    factors_combined = pd.concat(factor_frames, axis=1)
    factors_combined.to_csv(processed_dir / "factors_combined.csv")

    sector_value_mean, sector_value_std = sector_mean(value_df, categorizations)
    sector_momentum_mean, sector_momentum_std = sector_mean(momentum_df, categorizations)
    sector_volatility_mean, sector_volatility_std = sector_mean(
        volatility_df, categorizations
    )
    sector_dividend_mean, sector_dividend_std = sector_mean(dividend_df, categorizations)
    sector_pairs = [
        ("momentum", sector_momentum_mean, sector_momentum_std),
        ("value", sector_value_mean, sector_value_std),
        ("volatility", sector_volatility_mean, sector_volatility_std),
        ("dividend", sector_dividend_mean, sector_dividend_std),
    ]
    if not growth_df.empty:
        sector_pairs.append(("growth", *sector_mean(growth_df, categorizations)))
    if not quality_df.empty:
        sector_pairs.append(("quality", *sector_mean(quality_df, categorizations)))

    for name, mean_df, std_df in sector_pairs:
        combined = pd.concat(
            {f"{s}_mean": mean_df[s] for s in mean_df.columns}
            | {f"{s}_std": std_df[s] for s in std_df.columns},
            axis=1,
        )
        combined.to_csv(processed_dir / f"sector_{name}.csv")

    print(f"Saved factor CSVs to {processed_dir}")


if __name__ == "__main__":
    factor_processing(load_price_data(), load_book_data(),load_dividend_data())
