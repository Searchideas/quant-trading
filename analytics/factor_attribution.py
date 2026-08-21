"""Factor attribution (placeholder for future work)."""

from __future__ import annotations

import pandas as pd


class FactorAttribution:
    """Stub: decompose portfolio returns by factor contribution."""

    def __init__(self, returns: pd.Series, factors: pd.DataFrame):
        self.returns = returns
        self.factors = factors

    def run(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Factor attribution is not implemented yet. "
            "Use analytics.performance.summarize for portfolio-level stats."
        )
