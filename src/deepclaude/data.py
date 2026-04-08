"""Data loading with LRU caching over trader-data parquet files.

All arrays are (T, N) float32.  Read-only, safe for concurrent access.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from deepclaude.config import DATA_DIR


@lru_cache(maxsize=16)
def _load_field(field: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a vbt_ready parquet and return (data, dates, symbols)."""
    path = DATA_DIR / "vbt_ready" / f"{field}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No data file for field '{field}' at {path}")
    df = pd.read_parquet(path)
    dates = df.index.values
    symbols = list(df.columns)
    values = df.values.astype(np.float32)
    return values, dates, symbols


def get(field: str, start: str | None = None, end: str | None = None) -> np.ndarray:
    """Load a market data field as (T, N) float32 array."""
    values, dates, _ = _load_field(field)
    if start is None and end is None:
        return values
    mask = np.ones(len(dates), dtype=bool)
    if start is not None:
        mask &= dates >= np.datetime64(start)
    if end is not None:
        mask &= dates <= np.datetime64(end)
    return values[mask]


def get_dates(field: str = "close") -> np.ndarray:
    """Return date index as datetime64[ns] array."""
    _, dates, _ = _load_field(field)
    return dates


def get_symbols(field: str = "close") -> list[str]:
    """Return list of ticker symbols."""
    _, _, symbols = _load_field(field)
    return symbols


@lru_cache(maxsize=4)
def get_benchmark(name: str) -> np.ndarray:
    """Load benchmark close prices as 1D float32 array."""
    path = DATA_DIR / f"{name}_close.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No benchmark file at {path}")
    df = pd.read_parquet(path)
    return df.iloc[:, 0].values.astype(np.float32)


@lru_cache(maxsize=8)
def _load_universe(name: str) -> pd.DataFrame:
    """Load universe membership history."""
    if name == "spx":
        path = DATA_DIR / "spx_history" / "spx_history.parquet"
    else:
        path = DATA_DIR / "universe" / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No universe file at {path}")
    return pd.read_parquet(path)


def get_universe(name: str, month: str) -> list[str]:
    """Return list of tickers in universe for a given month."""
    df = _load_universe(name)
    filtered = df[df["month"] == month]
    return sorted(filtered["ticker"].tolist())


@lru_cache(maxsize=4)
def get_universe_mask(name: str) -> np.ndarray:
    """Return (T, N) bool mask: True if stock is in universe that month.

    Uses the date index and symbol list from the 'close' field.
    Membership is looked up monthly — all days in a month share the same mask.
    """
    _, dates, symbols = _load_field("close")
    T, N = len(dates), len(symbols)
    sym_to_idx = {s: i for i, s in enumerate(symbols)}

    df = _load_universe(name)
    # Build a dict: month_str -> set of ticker indices
    month_members: dict[str, set[int]] = {}
    for _, row in df.iterrows():
        m = row["month"]
        ticker = row["ticker"]
        if ticker in sym_to_idx:
            month_members.setdefault(m, set()).add(sym_to_idx[ticker])

    mask = np.zeros((T, N), dtype=bool)
    for t in range(T):
        month_str = str(dates[t])[:7]  # "YYYY-MM"
        members = month_members.get(month_str, set())
        for idx in members:
            mask[t, idx] = True

    return mask
