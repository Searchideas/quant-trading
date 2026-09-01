import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
def test_factor_weight_sum_to_one():
    total = (
        config.MOMENTUM_WEIGHT +
        config.VALUE_WEIGHT +
        config.VOLATILITY_WEIGHT +
        config.DIVIDEND_WEIGHT
    )
    assert total == 1.0
    