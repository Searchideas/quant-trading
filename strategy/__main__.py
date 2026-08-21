import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy import run_strategy

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run factor strategy")
    parser.add_argument("--market", default=None, choices=["us", "sg"])
    args = parser.parse_args()
    run_strategy(market=args.market)
