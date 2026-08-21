"""Filesystem helpers for market-scoped data directories."""

from __future__ import annotations

from pathlib import Path

from config import ROOT, get_paths, normalize_market

__all__ = ["ROOT", "ensure_market_dirs", "get_paths", "normalize_market"]


def ensure_market_dirs(market: str | None = None) -> tuple[Path, Path]:
    raw_dir, processed_dir = get_paths(market)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, processed_dir
