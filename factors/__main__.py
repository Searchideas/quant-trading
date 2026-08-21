import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factors import factor_processing, load_book_data, load_price_data

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute factors")
    parser.add_argument("--market", default=None, choices=["us", "sg"])
    args = parser.parse_args()
    factor_processing(
        load_price_data(args.market),
        load_book_data(args.market),
        market=args.market,
    )
