from pathlib import Path

import pandas as pd

from .io import load_book_data, load_categorizations, load_price_data
from .momentum import MomentumFactor
from .value import ValueFactor
from .volatility import VolatilityFactor

__all__ = [
    "ValueFactor",
    "MomentumFactor",
    "VolatilityFactor",
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

    value_df.to_csv(processed_dir / "value_data.csv")
    momentum_df.to_csv(processed_dir / "momentum_data.csv")
    volatility_df.to_csv(processed_dir / "volatility_data.csv")

    factors_combined = pd.concat(
        [
            value_df.add_suffix("_value"),
            momentum_df.add_suffix("_momentum"),
            volatility_df.add_suffix("_volatility"),
        ],
        axis=1,
    )
    factors_combined.to_csv(processed_dir / "factors_combined.csv")

    sector_value_mean, sector_value_std = sector_mean(value_df, categorizations)
    sector_momentum_mean, sector_momentum_std = sector_mean(momentum_df, categorizations)
    sector_volatility_mean, sector_volatility_std = sector_mean(
        volatility_df, categorizations
    )

    for name, mean_df, std_df in [
        ("momentum", sector_momentum_mean, sector_momentum_std),
        ("value", sector_value_mean, sector_value_std),
        ("volatility", sector_volatility_mean, sector_volatility_std),
    ]:
        combined = pd.concat(
            {f"{s}_mean": mean_df[s] for s in mean_df.columns}
            | {f"{s}_std": std_df[s] for s in std_df.columns},
            axis=1,
        )
        combined.to_csv(processed_dir / f"sector_{name}.csv")

    print(f"Saved factor CSVs to {processed_dir}")


if __name__ == "__main__":
    factor_processing(load_price_data(), load_book_data())
