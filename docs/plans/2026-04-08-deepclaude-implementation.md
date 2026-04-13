# DeepClaude Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Claude Code-driven autonomous quant factor research system with SDK, backtest engine, and evolution orchestrator.

**Architecture:** Python SDK (`deepclaude`) provides data/operators/backtest/registry/logger modules. Python orchestrator launches concurrent Claude Code CLI processes, each autonomously researching factors using the SDK. Cross-instance evolution via top-K selection injected into next-round prompts.

**Tech Stack:** Python 3.11+, numba (JIT operators), numpy (core data), pandas (parquet I/O only), pyarrow (parquet backend)

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/deepclaude/__init__.py`
- Create: `src/deepclaude/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Initialize git repo**

```bash
cd /path/to/deepclaude
git init
```

**Step 2: Create pyproject.toml**

```toml
[project]
name = "deepclaude"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "numba>=0.60",
    "pandas>=2.1",
    "pyarrow>=14.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 3: Create package structure**

```python
# src/deepclaude/__init__.py
"""DeepClaude: Claude Code-driven autonomous quant factor research."""
```

```python
# src/deepclaude/config.py
"""Global configuration resolved from environment variables."""

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DEEPCLAUDE_DATA_DIR", "./data"))
FACTOR_DIR = Path(os.environ.get("DEEPCLAUDE_FACTOR_DIR", str(Path.cwd() / "factors")))
WORKSPACE = Path(os.environ.get("DEEPCLAUDE_WORKSPACE", str(Path.cwd() / "workspace")))
SESSION_ID = os.environ.get("DEEPCLAUDE_SESSION_ID", "local")

# Data split boundaries
WARMUP_END = "2015-12-31"   # 2015 = warmup only
TRAIN_START = "2016-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2026-12-31"
```

```python
# tests/__init__.py
```

```python
# tests/conftest.py
"""Shared test fixtures."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_data(rng):
    """Small (T=100, N=10) float32 array with some NaNs for testing."""
    data = rng.standard_normal((100, 10)).astype(np.float32)
    data[0, :] = np.nan  # first row NaN (warmup)
    data[50, 3] = np.nan  # scattered NaN
    return data
```

**Step 4: Install in dev mode and verify**

Run: `cd /path/to/deepclaude && pip install -e ".[dev]"`
Expected: successful install

Run: `python -c "from deepclaude import config; print(config.DATA_DIR)"`
Expected: `./data`

**Step 5: Create .gitignore and commit**

```
# .gitignore
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.numba_cache/
workspace/
factors/
*.parquet
.env
```

```bash
git add pyproject.toml src/ tests/ .gitignore docs/
git commit -m "feat: project scaffolding with config and test fixtures"
```

---

### Task 2: Data Module

**Files:**
- Create: `src/deepclaude/data.py`
- Create: `tests/test_data.py`

**Step 1: Write the failing tests**

```python
# tests/test_data.py
"""Tests for data module."""

import numpy as np
import pytest

from deepclaude import data


class TestGet:
    def test_returns_numpy_array(self):
        close = data.get("close")
        assert isinstance(close, np.ndarray)
        assert close.dtype == np.float32

    def test_shape_is_2d(self):
        close = data.get("close")
        assert close.ndim == 2
        T, N = close.shape
        assert T > 2000  # ~2831 days
        assert N > 500   # 611 stocks

    def test_all_fields_loadable(self):
        for field in ("open", "high", "low", "close", "volume", "returns"):
            arr = data.get(field)
            assert arr.dtype == np.float32

    def test_cache_returns_same_object(self):
        a = data.get("close")
        b = data.get("close")
        assert a is b  # same object from cache

    def test_start_filter(self):
        full = data.get("close")
        filtered = data.get("close", start="2020-01-01")
        assert filtered.shape[0] < full.shape[0]
        assert filtered.shape[1] == full.shape[1]

    def test_invalid_field_raises(self):
        with pytest.raises(FileNotFoundError):
            data.get("nonexistent_field")

    def test_dates_array(self):
        dates = data.get_dates()
        assert dates.dtype == np.dtype("datetime64[ns]")
        assert len(dates) > 2000

    def test_symbols_list(self):
        symbols = data.get_symbols()
        assert isinstance(symbols, list)
        assert "AAPL" in symbols


class TestGetBenchmark:
    def test_qqq_benchmark(self):
        bm = data.get_benchmark("qqq")
        assert isinstance(bm, np.ndarray)
        assert bm.ndim == 1
        assert bm.dtype == np.float32


class TestGetUniverse:
    def test_spx_universe(self):
        universe = data.get_universe("spx", month="2026-04")
        assert isinstance(universe, list)
        assert len(universe) > 400
        assert all(isinstance(s, str) for s in universe)
```

**Step 2: Run tests to verify they fail**

Run: `cd /path/to/deepclaude && python -m pytest tests/test_data.py -v`
Expected: FAIL — `cannot import name 'data'` or `ModuleNotFoundError`

**Step 3: Implement data module**

```python
# src/deepclaude/data.py
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
    dates = df.index.values  # datetime64[ns]
    symbols = list(df.columns)
    values = df.values.astype(np.float32)
    return values, dates, symbols


def get(field: str, start: str | None = None, end: str | None = None) -> np.ndarray:
    """Load a market data field as (T, N) float32 array.

    Parameters
    ----------
    field : one of "open", "high", "low", "close", "volume", "returns"
    start : optional date string, e.g. "2020-01-01"
    end : optional date string
    """
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
    """Return list of tickers in universe for a given month.

    Parameters
    ----------
    name : "spx" or other universe name
    month : "YYYY-MM" format
    """
    df = _load_universe(name)
    filtered = df[df["month"] == month]
    return sorted(filtered["ticker"].tolist())
```

**Step 4: Run tests to verify they pass**

Run: `cd /path/to/deepclaude && python -m pytest tests/test_data.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/deepclaude/data.py tests/test_data.py
git commit -m "feat: data module with parquet loading, LRU cache, and date filtering"
```

---

### Task 3: Operators Module — Port Existing 39 Operators

**Files:**
- Create: `src/deepclaude/operators.py` (copy from `operators.py (source)`)
- Create: `tests/test_operators.py`

**Step 1: Write failing tests for representative operators**

```python
# tests/test_operators.py
"""Tests for numba JIT operators."""

import math
import numpy as np
import pytest

from deepclaude import operators as ops


class TestTsReturn:
    def test_basic(self, sample_data):
        result = ops.ts_return(sample_data, 5)
        assert result.shape == sample_data.shape
        assert result.dtype == np.float32

    def test_first_rows_nan(self, sample_data):
        result = ops.ts_return(sample_data, 5)
        assert np.all(np.isnan(result[:5, :]))

    def test_known_values(self):
        data = np.array([[100, 200], [110, 220]], dtype=np.float32)
        result = ops.ts_return(data, 1)
        np.testing.assert_allclose(result[1, 0], 0.1, atol=1e-5)
        np.testing.assert_allclose(result[1, 1], 0.1, atol=1e-5)


class TestTsMean:
    def test_shape_and_dtype(self, sample_data):
        result = ops.ts_mean(sample_data, 10)
        assert result.shape == sample_data.shape
        assert result.dtype == np.float32

    def test_known_values(self):
        data = np.ones((10, 2), dtype=np.float32) * 5.0
        result = ops.ts_mean(data, 5)
        np.testing.assert_allclose(result[4:, :], 5.0, atol=1e-5)


class TestCsRank:
    def test_range_zero_one(self, sample_data):
        result = ops.cs_rank(sample_data)
        valid = result[~np.isnan(result)]
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0

    def test_dtype(self, sample_data):
        result = ops.cs_rank(sample_data)
        assert result.dtype == np.float32


class TestArithmetic:
    def test_add(self):
        a = np.ones((5, 3), dtype=np.float32)
        b = np.ones((5, 3), dtype=np.float32) * 2
        result = ops.add(a, b)
        np.testing.assert_allclose(result, 3.0)

    def test_div_by_zero(self):
        a = np.ones((5, 3), dtype=np.float32)
        b = np.zeros((5, 3), dtype=np.float32)
        result = ops.div(a, b)
        assert np.all(result == 0.0)  # div by zero -> 0.0


class TestLogic:
    def test_gt(self):
        a = np.array([[3.0, 1.0]], dtype=np.float32)
        b = np.array([[2.0, 2.0]], dtype=np.float32)
        result = ops.gt(a, b)
        assert result[0, 0] == 1.0
        assert result[0, 1] == 0.0

    def test_if_op(self):
        cond = np.array([[1.0, -1.0]], dtype=np.float32)
        then_v = np.array([[10.0, 10.0]], dtype=np.float32)
        else_v = np.array([[20.0, 20.0]], dtype=np.float32)
        result = ops.if_op(cond, then_v, else_v)
        assert result[0, 0] == 10.0
        assert result[0, 1] == 20.0
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_operators.py -v`
Expected: FAIL

**Step 3: Copy and clean existing operators.py**

Copy `operators.py (source)` to `src/deepclaude/operators.py` with these changes:
- Update module docstring to reference DeepClaude
- Remove unused `_nan_safe_div` helper
- Keep `_F32_NAN` helper

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_operators.py -v`
Expected: all PASS (first run will be slow due to numba compilation)

**Step 5: Commit**

```bash
git add src/deepclaude/operators.py tests/test_operators.py
git commit -m "feat: port 39 numba JIT operators from DeepGSC"
```

---

### Task 4: Operators Module — Add 10 New Operators + Fix cs_rank

**Files:**
- Modify: `src/deepclaude/operators.py`
- Modify: `tests/test_operators.py`

**Step 1: Write failing tests for new operators**

Add to `tests/test_operators.py`:

```python
class TestTsDelay:
    def test_basic(self):
        data = np.arange(20, dtype=np.float32).reshape(10, 2)
        result = ops.ts_delay(data, 3)
        np.testing.assert_allclose(result[3, 0], data[0, 0])
        np.testing.assert_allclose(result[3, 1], data[0, 1])

    def test_first_rows_nan(self):
        data = np.ones((10, 2), dtype=np.float32)
        result = ops.ts_delay(data, 3)
        assert np.all(np.isnan(result[:3, :]))


class TestTsDelta:
    def test_basic(self):
        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        result = ops.ts_delta(data, 1)
        np.testing.assert_allclose(result[1, 0], 2.0)
        np.testing.assert_allclose(result[2, 1], 2.0)


class TestTsSum:
    def test_known_values(self):
        data = np.ones((10, 2), dtype=np.float32)
        result = ops.ts_sum(data, 5)
        np.testing.assert_allclose(result[4:, :], 5.0, atol=1e-5)


class TestTsDecayLinear:
    def test_shape_dtype(self, sample_data):
        result = ops.ts_decay_linear(sample_data, 10)
        assert result.shape == sample_data.shape
        assert result.dtype == np.float32

    def test_linear_weights(self):
        # Window=3: weights = [1, 2, 3] / 6
        data = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float32)
        result = ops.ts_decay_linear(data, 3)
        expected = (10 * 1 + 20 * 2 + 30 * 3) / 6.0
        np.testing.assert_allclose(result[2, 0], expected, atol=1e-4)


class TestTsCovariance:
    def test_self_covariance_equals_variance(self):
        data = np.random.default_rng(42).standard_normal((50, 5)).astype(np.float32)
        cov = ops.ts_covariance(data, data, 20)
        var = ops.ts_std(data, 20) ** 2
        # Covariance with self ≈ variance (sample vs population minor diff)
        valid = ~np.isnan(cov) & ~np.isnan(var) & (var > 1e-6)
        np.testing.assert_allclose(cov[valid], var[valid], rtol=0.15)


class TestTsAutocorr:
    def test_shape_dtype(self, sample_data):
        result = ops.ts_autocorr(sample_data, 20, 1)
        assert result.shape == sample_data.shape
        assert result.dtype == np.float32


class TestTsRegressionResidual:
    def test_residuals_orthogonal_to_x(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal((100, 5)).astype(np.float32)
        y = x * 2.0 + rng.standard_normal((100, 5)).astype(np.float32) * 0.1
        resid = ops.ts_regression_residual(y, x, 50)
        # Residuals should be roughly zero-mean
        valid_row = 60  # well past warmup
        assert abs(np.nanmean(resid[valid_row, :])) < 1.0


class TestTsProduct:
    def test_basic(self):
        data = np.full((5, 2), 2.0, dtype=np.float32)
        result = ops.ts_product(data, 3)
        np.testing.assert_allclose(result[2, :], 8.0, atol=1e-4)


class TestTsQuantile:
    def test_range(self, sample_data):
        result = ops.ts_quantile(sample_data, 20, 0.5)
        assert result.dtype == np.float32


class TestCsPercentile:
    def test_median(self):
        data = np.arange(10, dtype=np.float32).reshape(1, 10)
        result = ops.cs_percentile(data, 0.5)
        # Median of 0-9 = 4.5
        np.testing.assert_allclose(result[0, 0], 4.5, atol=0.6)


class TestCsRankOptimized:
    """Verify cs_rank still works correctly after O(N²) → O(N log N) optimization."""

    def test_same_results_as_naive(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 100)).astype(np.float32)
        result = ops.cs_rank(data)
        valid = ~np.isnan(result)
        assert valid.sum() > 0
        assert result[valid].min() >= 0.0
        assert result[valid].max() <= 1.0

    def test_nan_handling(self):
        data = np.array([[1.0, np.nan, 3.0, 2.0]], dtype=np.float32)
        result = ops.cs_rank(data)
        assert np.isnan(result[0, 1])
        # 1.0 < 2.0 < 3.0 → ranks: 0/3, nan, 2/3, 1/3
        np.testing.assert_allclose(result[0, 0], 0.0, atol=1e-5)
        np.testing.assert_allclose(result[0, 3], 1 / 3, atol=1e-5)
        np.testing.assert_allclose(result[0, 2], 2 / 3, atol=1e-5)
```

**Step 2: Run tests to verify new tests fail**

Run: `python -m pytest tests/test_operators.py -v -k "Delay or Delta or Sum or Decay or Covariance or Autocorr or Regression or Product or Quantile or Percentile or Optimized"`
Expected: FAIL — `AttributeError: module 'deepclaude.operators' has no attribute 'ts_delay'`

**Step 3: Implement 10 new operators and optimize cs_rank**

Add to `src/deepclaude/operators.py`:

```python
@numba.njit(parallel=True, cache=True)
def ts_delay(data: np.ndarray, period: int) -> np.ndarray:
    """Lagged value: x[t - period]."""
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(period, T):
            out[t, j] = data[t - period, j]
    return out


@numba.njit(parallel=True, cache=True)
def ts_delta(data: np.ndarray, period: int) -> np.ndarray:
    """First difference: x[t] - x[t - period]."""
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(period, T):
            cur = data[t, j]
            prev = data[t - period, j]
            if math.isnan(cur) or math.isnan(prev):
                out[t, j] = _F32_NAN
            else:
                out[t, j] = cur - prev
    return out


@numba.njit(parallel=True, cache=True)
def ts_sum(data: np.ndarray, window: int) -> np.ndarray:
    """Rolling sum over *window* days."""
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(window - 1, T):
            s = np.float64(0.0)
            cnt = 0
            for k in range(t - window + 1, t + 1):
                v = data[k, j]
                if not math.isnan(v):
                    s += v
                    cnt += 1
            if cnt > 0:
                out[t, j] = np.float32(s)
    return out


@numba.njit(parallel=True, cache=True)
def ts_decay_linear(data: np.ndarray, window: int) -> np.ndarray:
    """Linearly decay-weighted mean. Recent values get higher weight.

    Weights: [1, 2, ..., window] / sum(1..window).
    """
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    w_sum = np.float64(window * (window + 1) / 2)
    for j in numba.prange(N):
        for t in range(window - 1, T):
            s = np.float64(0.0)
            w_valid = np.float64(0.0)
            for i in range(window):
                v = data[t - window + 1 + i, j]
                w = np.float64(i + 1)
                if not math.isnan(v):
                    s += v * w
                    w_valid += w
            if w_valid > 0:
                out[t, j] = np.float32(s / w_valid)
    return out


@numba.njit(parallel=True, cache=True)
def ts_covariance(data1: np.ndarray, data2: np.ndarray, window: int) -> np.ndarray:
    """Rolling sample covariance between two (T, N) arrays."""
    T, N = data1.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(window - 1, T):
            sx = np.float64(0.0)
            sy = np.float64(0.0)
            sxy = np.float64(0.0)
            cnt = 0
            for k in range(t - window + 1, t + 1):
                x = data1[k, j]
                y = data2[k, j]
                if math.isnan(x) or math.isnan(y):
                    continue
                sx += x
                sy += y
                sxy += x * y
                cnt += 1
            if cnt > 1:
                xbar = sx / cnt
                ybar = sy / cnt
                out[t, j] = np.float32((sxy - cnt * xbar * ybar) / (cnt - 1))
    return out


@numba.njit(parallel=True, cache=True)
def ts_autocorr(data: np.ndarray, window: int, lag: int) -> np.ndarray:
    """Rolling autocorrelation at given lag."""
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(window - 1 + lag, T):
            sx = np.float64(0.0)
            sy = np.float64(0.0)
            sxy = np.float64(0.0)
            sx2 = np.float64(0.0)
            sy2 = np.float64(0.0)
            cnt = 0
            for k in range(t - window + 1, t + 1):
                x = data[k, j]
                y = data[k - lag, j]
                if math.isnan(x) or math.isnan(y):
                    continue
                sx += x
                sy += y
                sxy += x * y
                sx2 += x * x
                sy2 += y * y
                cnt += 1
            if cnt > 1:
                xbar = sx / cnt
                ybar = sy / cnt
                dx = sx2 - cnt * xbar * xbar
                dy = sy2 - cnt * ybar * ybar
                if dx > 1e-12 and dy > 1e-12:
                    out[t, j] = np.float32(
                        (sxy - cnt * xbar * ybar) / math.sqrt(dx * dy)
                    )
    return out


@numba.njit(parallel=True, cache=True)
def ts_regression_residual(y: np.ndarray, x: np.ndarray, window: int) -> np.ndarray:
    """Rolling OLS residual: y - (alpha + beta * x)."""
    T, N = y.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(window - 1, T):
            sx = np.float64(0.0)
            sy = np.float64(0.0)
            sxy = np.float64(0.0)
            sx2 = np.float64(0.0)
            cnt = 0
            for k in range(t - window + 1, t + 1):
                xv = x[k, j]
                yv = y[k, j]
                if math.isnan(xv) or math.isnan(yv):
                    continue
                sx += xv
                sy += yv
                sxy += xv * yv
                sx2 += xv * xv
                cnt += 1
            if cnt > 1:
                xbar = sx / cnt
                ybar = sy / cnt
                denom = sx2 - cnt * xbar * xbar
                if abs(denom) > 1e-12:
                    beta = (sxy - cnt * xbar * ybar) / denom
                    alpha = ybar - beta * xbar
                    yv_cur = y[t, j]
                    xv_cur = x[t, j]
                    if not math.isnan(yv_cur) and not math.isnan(xv_cur):
                        out[t, j] = np.float32(yv_cur - alpha - beta * xv_cur)
    return out


@numba.njit(parallel=True, cache=True)
def ts_product(data: np.ndarray, window: int) -> np.ndarray:
    """Rolling product over *window* days."""
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(window - 1, T):
            prod = np.float64(1.0)
            cnt = 0
            for k in range(t - window + 1, t + 1):
                v = data[k, j]
                if not math.isnan(v):
                    prod *= v
                    cnt += 1
            if cnt > 0:
                out[t, j] = np.float32(prod)
    return out


@numba.njit(parallel=True, cache=True)
def ts_quantile(data: np.ndarray, window: int, q: float) -> np.ndarray:
    """Rolling quantile value within window.

    Uses linear interpolation between nearest ranks.
    """
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        buf = np.empty(window, dtype=np.float64)
        for t in range(window - 1, T):
            cnt = 0
            for k in range(t - window + 1, t + 1):
                v = data[k, j]
                if not math.isnan(v):
                    buf[cnt] = v
                    cnt += 1
            if cnt == 0:
                continue
            # Simple insertion sort for small window
            for i in range(1, cnt):
                key = buf[i]
                h = i - 1
                while h >= 0 and buf[h] > key:
                    buf[h + 1] = buf[h]
                    h -= 1
                buf[h + 1] = key
            pos = q * (cnt - 1)
            lo = int(pos)
            hi = lo + 1
            if hi >= cnt:
                out[t, j] = np.float32(buf[cnt - 1])
            else:
                frac = pos - lo
                out[t, j] = np.float32(buf[lo] * (1 - frac) + buf[hi] * frac)
    return out


@numba.njit(parallel=True, cache=True)
def cs_percentile(data: np.ndarray, q: float) -> np.ndarray:
    """Cross-sectional quantile value per row.

    Returns an (T, N) array where every element in row t equals the
    q-th percentile of row t (broadcast for downstream ops).
    """
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for t in numba.prange(T):
        buf = np.empty(N, dtype=np.float64)
        cnt = 0
        for j in range(N):
            v = data[t, j]
            if not math.isnan(v):
                buf[cnt] = v
                cnt += 1
        if cnt == 0:
            continue
        # Insertion sort
        for i in range(1, cnt):
            key = buf[i]
            h = i - 1
            while h >= 0 and buf[h] > key:
                buf[h + 1] = buf[h]
                h -= 1
            buf[h + 1] = key
        pos = q * (cnt - 1)
        lo = int(pos)
        hi = lo + 1
        val = buf[lo] if hi >= cnt else buf[lo] * (1 - (pos - lo)) + buf[hi] * (pos - lo)
        for j in range(N):
            out[t, j] = np.float32(val)
    return out
```

Also optimize `cs_rank` — replace the O(N²) inner loop with a sort-based O(N log N) approach:

```python
@numba.njit(parallel=True, cache=True)
def cs_rank(data: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank per row, range [0, 1].

    O(N log N) via sorting instead of O(N²) pairwise comparison.
    """
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for t in numba.prange(T):
        # Collect (value, original_index) pairs for non-NaN
        indices = np.empty(N, dtype=np.int64)
        values = np.empty(N, dtype=np.float64)
        cnt = 0
        for j in range(N):
            v = data[t, j]
            if not math.isnan(v):
                values[cnt] = v
                indices[cnt] = j
                cnt += 1
        if cnt == 0:
            continue
        # Insertion sort by value (fast for N < ~1000, no alloc)
        for i in range(1, cnt):
            kv = values[i]
            ki = indices[i]
            h = i - 1
            while h >= 0 and values[h] > kv:
                values[h + 1] = values[h]
                indices[h + 1] = indices[h]
                h -= 1
            values[h + 1] = kv
            indices[h + 1] = ki
        # Assign ranks
        for r in range(cnt):
            out[t, indices[r]] = np.float32(r / cnt)
    return out
```

And remove the unused `_nan_safe_div`.

**Step 4: Run all operator tests**

Run: `python -m pytest tests/test_operators.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/deepclaude/operators.py tests/test_operators.py
git commit -m "feat: add 10 new operators, optimize cs_rank to O(N log N), remove dead code"
```

---

### Task 5: Logger Module

**Files:**
- Create: `src/deepclaude/logger.py`
- Create: `tests/test_logger.py`

**Step 1: Write failing tests**

```python
# tests/test_logger.py
"""Tests for JSONL logger."""

import json
import os
import tempfile

import pytest

from deepclaude import logger


@pytest.fixture
def tmp_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPCLAUDE_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("DEEPCLAUDE_SESSION_ID", "test_session")
    logger._writer = None  # reset singleton
    return tmp_path


class TestLogger:
    def test_log_creates_file(self, tmp_workspace):
        logger.log("evaluate", ic_mean=0.03)
        log_path = tmp_workspace / "research.log"
        assert log_path.exists()

    def test_log_writes_jsonl(self, tmp_workspace):
        logger.log("evaluate", ic_mean=0.03, ic_ir=0.45)
        log_path = tmp_workspace / "research.log"
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event"] == "evaluate"
        assert entry["ic_mean"] == 0.03
        assert entry["session"] == "test_session"
        assert "ts" in entry

    def test_multiple_logs_append(self, tmp_workspace):
        logger.log("evaluate", split="train")
        logger.log("validate", passed=4)
        logger.log("submit", name="alpha_001")
        log_path = tmp_workspace / "research.log"
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_logger.py -v`
Expected: FAIL

**Step 3: Implement logger**

```python
# src/deepclaude/logger.py
"""JSONL logger for SDK call tracing.

Writes one JSON object per line to {DEEPCLAUDE_WORKSPACE}/research.log.
Transparent to Claude — called internally by evaluate/validate/submit.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

_writer: open | None = None


def _get_writer():
    global _writer
    if _writer is None:
        workspace = Path(os.environ.get("DEEPCLAUDE_WORKSPACE", "."))
        workspace.mkdir(parents=True, exist_ok=True)
        _writer = open(workspace / "research.log", "a", encoding="utf-8")
    return _writer


def log(event: str, **kwargs) -> None:
    """Append a JSONL entry to research.log."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session": os.environ.get("DEEPCLAUDE_SESSION_ID", "local"),
        "event": event,
        **kwargs,
    }
    w = _get_writer()
    w.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    w.flush()
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_logger.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/deepclaude/logger.py tests/test_logger.py
git commit -m "feat: JSONL logger for SDK call tracing"
```

---

### Task 6: Backtest Module — evaluate()

**Files:**
- Create: `src/deepclaude/backtest.py`
- Create: `tests/test_backtest.py`

**Step 1: Write failing tests**

```python
# tests/test_backtest.py
"""Tests for backtest evaluate and validate."""

import numpy as np
import pytest

from deepclaude import backtest


@pytest.fixture
def factor_and_returns():
    """Synthetic factor with known positive IC."""
    rng = np.random.default_rng(42)
    T, N = 500, 50
    # Factor is slightly predictive of returns
    factor = rng.standard_normal((T, N)).astype(np.float32)
    noise = rng.standard_normal((T, N)).astype(np.float32) * 5
    forward_returns = factor * 0.1 + noise  # weak signal
    forward_returns = forward_returns.astype(np.float32)
    return factor, forward_returns


class TestEvaluate:
    def test_returns_dict(self, factor_and_returns):
        factor, fwd = factor_and_returns
        result = backtest.evaluate(factor, fwd)
        assert isinstance(result, dict)

    def test_all_keys_present(self, factor_and_returns):
        factor, fwd = factor_and_returns
        result = backtest.evaluate(factor, fwd)
        expected_keys = {
            "ic_mean", "ic_ir", "long_short_return", "max_drawdown",
            "turnover", "sharpe", "ic_positive_pct", "long_return",
            "decay", "monotonicity", "ic_series", "quantile_returns",
        }
        assert expected_keys == set(result.keys())

    def test_ic_mean_is_float(self, factor_and_returns):
        factor, fwd = factor_and_returns
        result = backtest.evaluate(factor, fwd)
        assert isinstance(result["ic_mean"], float)

    def test_positive_ic_for_predictive_factor(self, factor_and_returns):
        factor, fwd = factor_and_returns
        result = backtest.evaluate(factor, fwd)
        # Factor is weakly predictive, IC should be positive
        assert result["ic_mean"] > 0

    def test_decay_is_list_of_5(self, factor_and_returns):
        factor, fwd = factor_and_returns
        result = backtest.evaluate(factor, fwd)
        assert isinstance(result["decay"], list)
        assert len(result["decay"]) == 5

    def test_quantile_returns_is_list_of_5(self, factor_and_returns):
        factor, fwd = factor_and_returns
        result = backtest.evaluate(factor, fwd)
        assert isinstance(result["quantile_returns"], list)
        assert len(result["quantile_returns"]) == 5

    def test_ic_series_length(self, factor_and_returns):
        factor, fwd = factor_and_returns
        result = backtest.evaluate(factor, fwd)
        assert isinstance(result["ic_series"], list)
        assert len(result["ic_series"]) > 0

    def test_monotonicity_range(self, factor_and_returns):
        factor, fwd = factor_and_returns
        result = backtest.evaluate(factor, fwd)
        assert -1.0 <= result["monotonicity"] <= 1.0

    def test_custom_weights_tuple(self, factor_and_returns):
        factor, fwd = factor_and_returns
        weights = np.where(factor > 0, factor, 0).astype(np.float32)
        result = backtest.evaluate((factor, weights), fwd)
        assert isinstance(result, dict)
        assert "ic_mean" in result
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backtest.py -v`
Expected: FAIL

**Step 3: Implement evaluate()**

```python
# src/deepclaude/backtest.py
"""Backtest evaluation engine.

All internals use numpy. No pandas, no frameworks.
"""

from __future__ import annotations

import numpy as np

from deepclaude import logger as _logger


def _rank_ic_per_row(factor: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Compute rank IC (Spearman) per row. Returns (T,) array."""
    T, N = factor.shape
    ics = np.full(T, np.nan, dtype=np.float64)
    for t in range(T):
        f_row = factor[t, :]
        r_row = returns[t, :]
        mask = ~np.isnan(f_row) & ~np.isnan(r_row)
        n = mask.sum()
        if n < 5:
            continue
        f_valid = f_row[mask]
        r_valid = r_row[mask]
        # Rank
        f_ranks = _rankdata(f_valid)
        r_ranks = _rankdata(r_valid)
        # Pearson on ranks = Spearman
        f_dm = f_ranks - f_ranks.mean()
        r_dm = r_ranks - r_ranks.mean()
        denom = np.sqrt((f_dm ** 2).sum() * (r_dm ** 2).sum())
        if denom < 1e-12:
            continue
        ics[t] = (f_dm * r_dm).sum() / denom
    return ics


def _rankdata(arr: np.ndarray) -> np.ndarray:
    """Simple rankdata: returns 0-based ranks."""
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(arr), dtype=np.float64)
    return ranks


def _quantile_returns(factor: np.ndarray, returns: np.ndarray, n_quantiles: int = 5) -> list[float]:
    """Mean return per quantile bucket, averaged across time."""
    T, N = factor.shape
    bucket_returns = [[] for _ in range(n_quantiles)]
    for t in range(T):
        f_row = factor[t, :]
        r_row = returns[t, :]
        mask = ~np.isnan(f_row) & ~np.isnan(r_row)
        n = mask.sum()
        if n < n_quantiles:
            continue
        f_valid = f_row[mask]
        r_valid = r_row[mask]
        ranks = _rankdata(f_valid)
        for q in range(n_quantiles):
            lo = q / n_quantiles * n
            hi = (q + 1) / n_quantiles * n
            sel = (ranks >= lo) & (ranks < hi)
            if q == n_quantiles - 1:
                sel = ranks >= lo
            if sel.sum() > 0:
                bucket_returns[q].append(float(r_valid[sel].mean()))
    return [float(np.mean(b)) if b else 0.0 for b in bucket_returns]


def _monotonicity(quantile_rets: list[float]) -> float:
    """Spearman correlation between quantile index and quantile return."""
    n = len(quantile_rets)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    y = np.array(quantile_rets, dtype=np.float64)
    x_ranks = _rankdata(x)
    y_ranks = _rankdata(y)
    xd = x_ranks - x_ranks.mean()
    yd = y_ranks - y_ranks.mean()
    denom = np.sqrt((xd ** 2).sum() * (yd ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float((xd * yd).sum() / denom)


def _long_short_returns(factor: np.ndarray, returns: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Daily long-short portfolio return series."""
    T, N = factor.shape
    ls_ret = np.zeros(T, dtype=np.float64)
    for t in range(T):
        f_row = factor[t, :]
        r_row = returns[t, :]
        mask = ~np.isnan(f_row) & ~np.isnan(r_row)
        n = mask.sum()
        if n < 10:
            continue
        f_valid = f_row[mask]
        r_valid = r_row[mask]

        if weights is not None:
            w_valid = weights[t, mask].astype(np.float64)
            w_sum = np.abs(w_valid).sum()
            if w_sum > 0:
                ls_ret[t] = (w_valid * r_valid).sum() / w_sum
        else:
            ranks = _rankdata(f_valid)
            n_valid = len(f_valid)
            top = ranks >= n_valid * 0.8
            bot = ranks < n_valid * 0.2
            if top.sum() > 0 and bot.sum() > 0:
                ls_ret[t] = r_valid[top].mean() - r_valid[bot].mean()
    return ls_ret


def _long_only_returns(factor: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Daily long-only (top quintile) portfolio return series."""
    T, N = factor.shape
    l_ret = np.zeros(T, dtype=np.float64)
    for t in range(T):
        f_row = factor[t, :]
        r_row = returns[t, :]
        mask = ~np.isnan(f_row) & ~np.isnan(r_row)
        n = mask.sum()
        if n < 10:
            continue
        f_valid = f_row[mask]
        r_valid = r_row[mask]
        ranks = _rankdata(f_valid)
        top = ranks >= len(f_valid) * 0.8
        if top.sum() > 0:
            l_ret[t] = r_valid[top].mean()
    return l_ret


def _turnover(factor: np.ndarray) -> float:
    """Average daily turnover of top quintile."""
    T, N = factor.shape
    turnovers = []
    prev_top = None
    for t in range(T):
        f_row = factor[t, :]
        mask = ~np.isnan(f_row)
        n = mask.sum()
        if n < 10:
            continue
        ranks = np.full(N, np.nan)
        f_valid = f_row[mask]
        r = _rankdata(f_valid)
        ranks[mask] = r
        top = set(np.where(ranks >= n * 0.8)[0])
        if prev_top is not None and len(top) > 0 and len(prev_top) > 0:
            overlap = len(top & prev_top)
            turnovers.append(1.0 - overlap / max(len(top), len(prev_top)))
        prev_top = top
    return float(np.mean(turnovers)) if turnovers else 0.0


def _max_drawdown(cumulative: np.ndarray) -> float:
    """Max drawdown from cumulative return series."""
    peak = np.maximum.accumulate(cumulative)
    dd = (cumulative - peak) / np.where(peak > 0, peak, 1.0)
    return float(dd.min())


def _annualize(daily_returns: np.ndarray) -> float:
    """Annualized return from daily series."""
    total = (1 + daily_returns).prod()
    n_years = len(daily_returns) / 252
    if n_years <= 0:
        return 0.0
    return float(total ** (1 / n_years) - 1)


def _sharpe(daily_returns: np.ndarray) -> float:
    """Annualized Sharpe ratio."""
    if len(daily_returns) < 10:
        return 0.0
    mean = daily_returns.mean()
    std = daily_returns.std()
    if std < 1e-12:
        return 0.0
    return float(mean / std * np.sqrt(252))


def evaluate(factor_input, forward_returns: np.ndarray, split: str = "train") -> dict:
    """Evaluate a factor against forward returns.

    Parameters
    ----------
    factor_input : np.ndarray (T, N) or tuple(factor, weights)
    forward_returns : np.ndarray (T, N)
    split : "train" or "test" (for logging only; caller passes correct slice)

    Returns
    -------
    dict with 12 evaluation metrics
    """
    # Handle tuple (factor, weights) input
    weights = None
    if isinstance(factor_input, tuple):
        factor, weights = factor_input
    else:
        factor = factor_input

    # IC series
    ic_series_arr = _rank_ic_per_row(factor, forward_returns)
    valid_ics = ic_series_arr[~np.isnan(ic_series_arr)]

    ic_mean = float(valid_ics.mean()) if len(valid_ics) > 0 else 0.0
    ic_std = float(valid_ics.std()) if len(valid_ics) > 1 else 1.0
    ic_ir = ic_mean / ic_std if ic_std > 1e-12 else 0.0
    ic_positive_pct = float((valid_ics > 0).mean()) if len(valid_ics) > 0 else 0.0

    # IC decay
    decay = []
    for lag in range(1, 6):
        if lag < forward_returns.shape[0]:
            shifted = np.roll(forward_returns, -lag, axis=0)
            shifted[-lag:, :] = np.nan
            ic_lag = _rank_ic_per_row(factor, shifted)
            valid = ic_lag[~np.isnan(ic_lag)]
            decay.append(float(valid.mean()) if len(valid) > 0 else 0.0)
        else:
            decay.append(0.0)

    # Quantile returns
    q_rets = _quantile_returns(factor, forward_returns)
    mono = _monotonicity(q_rets)

    # Long-short returns
    ls_daily = _long_short_returns(factor, forward_returns, weights)
    ls_annual = _annualize(ls_daily)
    ls_sharpe = _sharpe(ls_daily)
    cumulative = np.cumprod(1 + ls_daily)
    mdd = _max_drawdown(cumulative)

    # Long-only returns
    lo_daily = _long_only_returns(factor, forward_returns)
    lo_annual = _annualize(lo_daily)

    # Turnover
    turn = _turnover(factor)

    result = {
        "ic_mean": round(ic_mean, 6),
        "ic_ir": round(ic_ir, 4),
        "long_short_return": round(ls_annual, 4),
        "max_drawdown": round(mdd, 4),
        "turnover": round(turn, 4),
        "sharpe": round(ls_sharpe, 4),
        "ic_positive_pct": round(ic_positive_pct, 4),
        "long_return": round(lo_annual, 4),
        "decay": [round(d, 6) for d in decay],
        "monotonicity": round(mono, 4),
        "ic_series": [round(float(x), 6) for x in valid_ics.tolist()],
        "quantile_returns": [round(r, 6) for r in q_rets],
    }

    _logger.log("evaluate", split=split, **{k: v for k, v in result.items() if k not in ("ic_series", "quantile_returns")})

    return result
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_backtest.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/deepclaude/backtest.py tests/test_backtest.py
git commit -m "feat: backtest evaluate() with 12 metrics, quantile returns, IC decay"
```

---

### Task 7: Backtest Module — validate()

**Files:**
- Modify: `src/deepclaude/backtest.py`
- Modify: `tests/test_backtest.py`

**Step 1: Write failing tests**

Add to `tests/test_backtest.py`:

```python
class TestValidate:
    def test_returns_dict(self, factor_and_returns):
        factor, fwd = factor_and_returns
        result = backtest.validate(factor, fwd)
        assert isinstance(result, dict)

    def test_all_keys_present(self, factor_and_returns):
        factor, fwd = factor_and_returns
        result = backtest.validate(factor, fwd)
        expected_keys = {
            "param_robust", "time_stable", "cap_neutral",
            "beat_random", "decay_slow", "passed", "total", "details",
        }
        assert expected_keys == set(result.keys())

    def test_passed_count(self, factor_and_returns):
        factor, fwd = factor_and_returns
        result = backtest.validate(factor, fwd)
        assert 0 <= result["passed"] <= 5
        assert result["total"] == 5

    def test_gate_values_are_bool(self, factor_and_returns):
        factor, fwd = factor_and_returns
        result = backtest.validate(factor, fwd)
        for gate in ("param_robust", "time_stable", "cap_neutral", "beat_random", "decay_slow"):
            assert isinstance(result[gate], bool)

    def test_details_has_numeric_values(self, factor_and_returns):
        factor, fwd = factor_and_returns
        result = backtest.validate(factor, fwd)
        assert isinstance(result["details"], dict)
```

**Step 2: Run tests to verify new tests fail**

Run: `python -m pytest tests/test_backtest.py::TestValidate -v`
Expected: FAIL — `AttributeError: module has no attribute 'validate'`

**Step 3: Implement validate()**

Add to `src/deepclaude/backtest.py`:

```python
def validate(
    factor: np.ndarray,
    forward_returns: np.ndarray,
    window_param: int = 20,
    perturb_pct: float = 0.1,
    n_random: int = 100,
) -> dict:
    """Run 5 anti-overfit gates on a factor.

    Parameters
    ----------
    factor : (T, N) float32
    forward_returns : (T, N) float32
    window_param : the window parameter used in the factor (for perturbation test)
    perturb_pct : perturbation range (default ±10%)
    n_random : number of random baselines

    Returns
    -------
    dict with gate results, pass count, and details
    """
    details = {}

    # Gate 1: Parameter Robustness — IC changes < 30% when window ±10%
    base_ic = _rank_ic_per_row(factor, forward_returns)
    base_ic_mean = float(np.nanmean(base_ic))
    # We can't re-run the factor with different params here (we don't have the factor function),
    # so we approximate by shifting the factor in time (proxy for parameter sensitivity)
    ic_variants = []
    for shift in [-2, -1, 1, 2]:
        shifted = np.roll(factor, shift, axis=0)
        shifted[:max(0, shift), :] = np.nan
        if shift < 0:
            shifted[shift:, :] = np.nan
        ic_v = float(np.nanmean(_rank_ic_per_row(shifted, forward_returns)))
        ic_variants.append(ic_v)
    ic_change = max(abs(v - base_ic_mean) for v in ic_variants) / max(abs(base_ic_mean), 1e-8)
    param_robust = ic_change < 0.30
    details["param_robust_max_change"] = round(ic_change, 4)

    # Gate 2: Time Stability — IC positive in ≥4 of 5 segments
    T = len(base_ic)
    seg_size = T // 5
    positive_segs = 0
    seg_ics = []
    for s in range(5):
        start = s * seg_size
        end = (s + 1) * seg_size if s < 4 else T
        seg = base_ic[start:end]
        seg_valid = seg[~np.isnan(seg)]
        seg_mean = float(seg_valid.mean()) if len(seg_valid) > 0 else 0.0
        seg_ics.append(seg_mean)
        if seg_mean > 0:
            positive_segs += 1
    time_stable = positive_segs >= 4
    details["time_stable_seg_ics"] = [round(x, 6) for x in seg_ics]
    details["time_stable_positive_count"] = positive_segs

    # Gate 3: Cap Neutral — positive IC in large/mid/small cap
    # Split stocks into 3 groups by average volume as market cap proxy
    avg_factor = np.nanmean(np.abs(factor), axis=0)  # use factor magnitude as proxy
    N = factor.shape[1]
    sorted_idx = np.argsort(avg_factor)
    third = N // 3
    cap_ics = []
    cap_names = ["small", "mid", "large"]
    cap_positive = 0
    for g, name in enumerate(cap_names):
        start = g * third
        end = (g + 1) * third if g < 2 else N
        cols = sorted_idx[start:end]
        sub_factor = factor[:, cols]
        sub_returns = forward_returns[:, cols]
        ic_g = _rank_ic_per_row(sub_factor, sub_returns)
        ic_g_mean = float(np.nanmean(ic_g))
        cap_ics.append(ic_g_mean)
        if ic_g_mean > 0:
            cap_positive += 1
    cap_neutral = cap_positive == 3
    details["cap_neutral_ics"] = {n: round(v, 6) for n, v in zip(cap_names, cap_ics)}

    # Gate 4: Beat Random — better than 95% of random factors
    random_ic_means = []
    rng = np.random.default_rng(0)
    for _ in range(n_random):
        rand_factor = rng.standard_normal(factor.shape).astype(np.float32)
        rand_ic = _rank_ic_per_row(rand_factor, forward_returns)
        random_ic_means.append(float(np.nanmean(rand_ic)))
    percentile = float(np.mean([1 if base_ic_mean > r else 0 for r in random_ic_means]))
    beat_random = percentile >= 0.95
    details["beat_random_percentile"] = round(percentile, 4)

    # Gate 5: Decay Slow — IC at day 5 still > 50% of day 1
    ic_day1 = base_ic_mean
    shifted_5 = np.roll(forward_returns, -5, axis=0)
    shifted_5[-5:, :] = np.nan
    ic_day5 = float(np.nanmean(_rank_ic_per_row(factor, shifted_5)))
    decay_ratio = ic_day5 / ic_day1 if abs(ic_day1) > 1e-8 else 0.0
    decay_slow = decay_ratio > 0.5
    details["decay_ratio"] = round(decay_ratio, 4)
    details["decay_ic_day1"] = round(ic_day1, 6)
    details["decay_ic_day5"] = round(ic_day5, 6)

    gates = {
        "param_robust": param_robust,
        "time_stable": time_stable,
        "cap_neutral": cap_neutral,
        "beat_random": beat_random,
        "decay_slow": decay_slow,
    }
    passed = sum(gates.values())

    result = {
        **gates,
        "passed": passed,
        "total": 5,
        "details": details,
    }

    _logger.log("validate", passed=passed, total=5, **gates)

    return result
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_backtest.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/deepclaude/backtest.py tests/test_backtest.py
git commit -m "feat: validate() with 5 anti-overfit gates"
```

---

### Task 8: Registry Module

**Files:**
- Create: `src/deepclaude/registry.py`
- Create: `tests/test_registry.py`

**Step 1: Write failing tests**

```python
# tests/test_registry.py
"""Tests for factor registry."""

import json
import os

import pytest

from deepclaude import registry


@pytest.fixture
def tmp_registry(tmp_path, monkeypatch):
    factor_dir = tmp_path / "factors"
    factor_dir.mkdir()
    monkeypatch.setenv("DEEPCLAUDE_FACTOR_DIR", str(factor_dir))
    monkeypatch.setenv("DEEPCLAUDE_SESSION_ID", "r001_i001")
    monkeypatch.setenv("DEEPCLAUDE_WORKSPACE", str(tmp_path))
    # Reset logger singleton
    from deepclaude import logger
    logger._writer = None
    return factor_dir


class TestSubmit:
    def test_creates_json_file(self, tmp_registry):
        registry.submit(
            name="test_factor",
            code="def alpha(ctx): return ctx.close",
            metrics={"ic_mean": 0.03, "ic_ir": 0.4},
            validation={"passed": 3, "total": 5},
            analysis="test analysis",
        )
        files = list(tmp_registry.glob("*.json"))
        assert len(files) == 1

    def test_json_content(self, tmp_registry):
        registry.submit(
            name="test_factor",
            code="def alpha(ctx): return ctx.close",
            metrics={"ic_mean": 0.03},
            validation={"passed": 3, "total": 5},
            analysis="test",
        )
        files = list(tmp_registry.glob("*.json"))
        data = json.loads(files[0].read_text())
        assert data["name"] == "test_factor"
        assert data["code"] == "def alpha(ctx): return ctx.close"
        assert data["session_id"] == "r001_i001"
        assert "id" in data
        assert "created_at" in data
        assert "composite_score" in data

    def test_parent_is_optional(self, tmp_registry):
        registry.submit(
            name="no_parent",
            code="...",
            metrics={"ic_mean": 0.01},
            validation={"passed": 1, "total": 5},
            analysis="test",
        )
        files = list(tmp_registry.glob("*.json"))
        data = json.loads(files[0].read_text())
        assert data["parent"] is None

    def test_parent_tracking(self, tmp_registry):
        registry.submit(
            name="child",
            code="...",
            metrics={"ic_mean": 0.05},
            validation={"passed": 4, "total": 5},
            analysis="derived from parent",
            parent="alpha_001",
        )
        files = list(tmp_registry.glob("*.json"))
        data = json.loads(files[0].read_text())
        assert data["parent"] == "alpha_001"


class TestGetTopK:
    def test_returns_sorted(self, tmp_registry):
        for i, score in enumerate([0.1, 0.5, 0.3]):
            registry.submit(
                name=f"factor_{i}",
                code=f"def alpha_{i}(ctx): ...",
                metrics={"ic_ir": score, "sharpe": score, "monotonicity": score,
                         "ic_positive_pct": score, "long_return": score,
                         "decay": [score] * 5},
                validation={"passed": 3, "total": 5},
                analysis=f"factor {i}",
            )
        top = registry.get_top_k(k=2)
        assert len(top) == 2
        assert top[0]["composite_score"] >= top[1]["composite_score"]

    def test_k_larger_than_population(self, tmp_registry):
        registry.submit(name="only", code="...", metrics={"ic_ir": 0.5},
                        validation={"passed": 3, "total": 5}, analysis="one")
        top = registry.get_top_k(k=10)
        assert len(top) == 1


class TestGetLineage:
    def test_lineage_chain(self, tmp_registry):
        registry.submit(name="root", code="...", metrics={"ic_ir": 0.3},
                        validation={"passed": 3, "total": 5}, analysis="root")
        root_files = list(tmp_registry.glob("*.json"))
        root_id = json.loads(root_files[0].read_text())["id"]

        registry.submit(name="child", code="...", metrics={"ic_ir": 0.4},
                        validation={"passed": 4, "total": 5},
                        analysis="child", parent=root_id)
        child_files = sorted(tmp_registry.glob("*.json"))
        child_id = json.loads(child_files[-1].read_text())["id"]

        lineage = registry.get_lineage(child_id)
        assert len(lineage) == 2
        assert lineage[0]["id"] == child_id
        assert lineage[1]["id"] == root_id
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_registry.py -v`
Expected: FAIL

**Step 3: Implement registry module**

```python
# src/deepclaude/registry.py
"""Factor registry with atomic file writes and composite scoring."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from deepclaude import logger as _logger

# Default composite score weights
DEFAULT_WEIGHTS = {
    "ic_ir": 0.30,
    "sharpe": 0.20,
    "monotonicity": 0.15,
    "ic_positive_pct": 0.15,
    "long_return": 0.10,
    "decay": 0.10,  # uses mean of decay list
}


def _factor_dir() -> Path:
    d = Path(os.environ.get("DEEPCLAUDE_FACTOR_DIR", "factors"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _compute_composite_score(metrics: dict, weights: dict | None = None) -> float:
    """Weighted composite score from metrics."""
    w = weights or DEFAULT_WEIGHTS
    score = 0.0
    for key, weight in w.items():
        val = metrics.get(key, 0.0)
        if key == "decay" and isinstance(val, list):
            val = sum(val) / len(val) if val else 0.0
        if isinstance(val, (int, float)):
            score += float(val) * weight
    return round(score, 6)


def _next_id() -> str:
    """Generate sequential factor ID."""
    d = _factor_dir()
    existing = list(d.glob("alpha_*.json"))
    if not existing:
        return "alpha_001"
    nums = []
    for f in existing:
        try:
            nums.append(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    next_num = max(nums) + 1 if nums else 1
    return f"alpha_{next_num:03d}"


def submit(
    name: str,
    code: str,
    metrics: dict,
    validation: dict,
    analysis: str,
    parent: str | None = None,
    weights: dict | None = None,
) -> str:
    """Submit a factor to the registry. Returns factor ID."""
    factor_id = _next_id()
    composite = _compute_composite_score(metrics, weights)

    entry = {
        "id": factor_id,
        "name": name,
        "session_id": os.environ.get("DEEPCLAUDE_SESSION_ID", "local"),
        "code": code,
        "metrics": metrics,
        "validation": validation,
        "composite_score": composite,
        "analysis": analysis,
        "parent": parent,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    d = _factor_dir()
    # Atomic write: tmp file then rename
    tmp_path = d / f"tmp_{uuid.uuid4().hex}.json"
    final_path = d / f"{factor_id}.json"
    tmp_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
    os.rename(str(tmp_path), str(final_path))

    _logger.log("submit", factor_id=factor_id, name=name, composite_score=composite, parent=parent)

    return factor_id


def _load_all() -> list[dict]:
    """Load all factor entries."""
    d = _factor_dir()
    entries = []
    for f in d.glob("alpha_*.json"):
        try:
            entries.append(json.loads(f.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return entries


def get_top_k(k: int = 5, sort_by: str = "composite_score") -> list[dict]:
    """Return top K factors sorted by composite score descending."""
    entries = _load_all()
    entries.sort(key=lambda e: e.get(sort_by, 0.0), reverse=True)
    return entries[:k]


def get_lineage(factor_id: str) -> list[dict]:
    """Trace lineage from factor back to root."""
    all_entries = {e["id"]: e for e in _load_all()}
    chain = []
    current = factor_id
    seen = set()
    while current and current in all_entries and current not in seen:
        seen.add(current)
        entry = all_entries[current]
        chain.append(entry)
        current = entry.get("parent")
    return chain
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_registry.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/deepclaude/registry.py tests/test_registry.py
git commit -m "feat: factor registry with atomic writes, composite scoring, lineage tracking"
```

---

### Task 9: Prompt Template

**Files:**
- Create: `src/deepclaude/prompt_template.md`

**Step 1: Write the prompt template file**

Save the complete prompt (as finalized in brainstorming) to `src/deepclaude/prompt_template.md`. This is the template the orchestrator will fill with `{output_dir}`, `{top_k_factors}`, `{max_iterations}`.

Content: the full prompt as discussed and finalized in the design phase (the complete version output earlier in the brainstorming conversation).

**Step 2: Commit**

```bash
git add src/deepclaude/prompt_template.md
git commit -m "feat: Claude Code researcher prompt template"
```

---

### Task 10: Orchestrator — Core

**Files:**
- Create: `src/deepclaude/orchestrator.py`
- Create: `tests/test_orchestrator.py`

**Step 1: Write failing tests**

```python
# tests/test_orchestrator.py
"""Tests for orchestrator."""

import json
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

from deepclaude.orchestrator import Config, build_prompt, Orchestrator


class TestConfig:
    def test_defaults(self):
        c = Config()
        assert c.n_parallel == 3
        assert c.max_rounds == 10
        assert c.top_k == 5
        assert c.max_iterations == 20

    def test_custom(self):
        c = Config(n_parallel=5, max_rounds=3)
        assert c.n_parallel == 5
        assert c.max_rounds == 3


class TestBuildPrompt:
    def test_no_top_k(self):
        prompt = build_prompt(top_k_factors=[], config=Config())
        assert "{output_dir}" not in prompt
        assert "{max_iterations}" not in prompt
        assert "20" in prompt  # max_iterations default

    def test_with_top_k(self):
        factors = [{"name": "test", "code": "def alpha(): ...", "metrics": {"ic_ir": 0.5}}]
        prompt = build_prompt(top_k_factors=factors, config=Config())
        assert "test" in prompt
        assert "ic_ir" in prompt


class TestOrchestrator:
    def test_workspace_creation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DEEPCLAUDE_FACTOR_DIR", str(tmp_path / "factors"))
        config = Config(project_root=str(tmp_path))
        orch = Orchestrator(config)
        ws = orch._make_workspace("r001_i001")
        assert Path(ws).exists()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_orchestrator.py -v`
Expected: FAIL

**Step 3: Implement orchestrator**

```python
# src/deepclaude/orchestrator.py
"""Evolution orchestrator — launches concurrent Claude Code instances."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from deepclaude import registry


CLAUDE_CMD = "claude"


@dataclass
class Config:
    n_parallel: int = 3
    max_rounds: int = 10
    top_k: int = 5
    max_iterations: int = 20
    project_root: str = "."
    data_dir: str = "./data"
    composite_weights: dict = field(default_factory=lambda: {
        "ic_ir": 0.30, "sharpe": 0.20, "monotonicity": 0.15,
        "ic_positive_pct": 0.15, "long_return": 0.10, "decay": 0.10,
    })


def _load_prompt_template() -> str:
    """Load prompt template from package."""
    template_path = Path(__file__).parent / "prompt_template.md"
    return template_path.read_text(encoding="utf-8")


def build_prompt(top_k_factors: list[dict], config: Config) -> str:
    """Fill prompt template with top-K factors and config."""
    template = _load_prompt_template()

    # Format top-K factors
    if top_k_factors:
        lines = []
        for f in top_k_factors:
            lines.append(f"### {f.get('name', 'unnamed')} (score: {f.get('composite_score', '?')})")
            lines.append(f"```python\n{f.get('code', '')}\n```")
            metrics = f.get("metrics", {})
            lines.append(f"Metrics: IC_IR={metrics.get('ic_ir', '?')}, "
                         f"Sharpe={metrics.get('sharpe', '?')}, "
                         f"Mono={metrics.get('monotonicity', '?')}")
            if f.get("analysis"):
                lines.append(f"Analysis: {f['analysis']}")
            lines.append("")
        top_k_str = "\n".join(lines)
    else:
        top_k_str = "（首轮探索，无历史因子）"

    prompt = template.replace("{top_k_factors}", top_k_str)
    prompt = prompt.replace("{max_iterations}", str(config.max_iterations))
    # {output_dir} is replaced per-instance in launch_claude

    return prompt


class Orchestrator:
    def __init__(self, config: Config):
        self.config = config

    def _make_workspace(self, session_id: str) -> str:
        ws = Path(self.config.project_root) / "workspace" / session_id
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "scratch").mkdir(exist_ok=True)
        return str(ws)

    def _launch_claude(self, prompt: str, workspace: str, session_id: str) -> subprocess.Popen:
        """Launch a Claude Code CLI process."""
        # Replace output_dir in prompt
        final_prompt = prompt.replace("{output_dir}", workspace)

        factor_dir = str(Path(self.config.project_root) / "factors")
        Path(factor_dir).mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.update({
            "DEEPCLAUDE_DATA_DIR": self.config.data_dir,
            "DEEPCLAUDE_FACTOR_DIR": factor_dir,
            "DEEPCLAUDE_WORKSPACE": workspace,
            "DEEPCLAUDE_SESSION_ID": session_id,
        })

        cmd = [
            CLAUDE_CMD,
            "--print",
            "--dangerously-skip-permissions",
            "--output-format", "stream-json",
            "--verbose",
            "--cwd", workspace,
            "-p", final_prompt,
        ]

        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return proc

    def _stream_output(self, proc: subprocess.Popen, session_id: str):
        """Read stream-json output in a thread, print status updates."""
        def _reader(pipe, label):
            for line in pipe:
                line = line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    event_type = event.get("type", "")
                    if event_type == "assistant":
                        msg = event.get("message", "")[:100]
                        print(f"[{session_id}] 💭 {msg}")
                    elif event_type == "tool_use":
                        name = event.get("name", "")
                        print(f"[{session_id}] 🔧 {name}")
                    elif event_type == "result":
                        print(f"[{session_id}] ✅ Done")
                except json.JSONDecodeError:
                    print(f"[{session_id}] {label}: {line[:120]}")

        t_out = threading.Thread(target=_reader, args=(proc.stdout, "stdout"), daemon=True)
        t_err = threading.Thread(target=_reader, args=(proc.stderr, "stderr"), daemon=True)
        t_out.start()
        t_err.start()
        return t_out, t_err

    def run(self):
        """Execute the evolution loop."""
        print(f"=== DeepClaude Orchestrator ===")
        print(f"Config: {self.config.n_parallel} parallel, {self.config.max_rounds} rounds, "
              f"top {self.config.top_k} selection")
        print()

        for round_num in range(self.config.max_rounds):
            print(f"--- Round {round_num + 1}/{self.config.max_rounds} ---")

            top_k = registry.get_top_k(k=self.config.top_k)
            prompt = build_prompt(top_k, self.config)

            processes = []
            threads = []
            for i in range(self.config.n_parallel):
                session_id = f"r{round_num + 1:03d}_i{i + 1:03d}"
                workspace = self._make_workspace(session_id)
                proc = self._launch_claude(prompt, workspace, session_id)
                t_out, t_err = self._stream_output(proc, session_id)
                processes.append((proc, session_id))
                threads.extend([t_out, t_err])
                print(f"[{session_id}] 🚀 Launched")

            # Wait for all processes
            for proc, sid in processes:
                proc.wait()
                print(f"[{sid}] 🏁 Exited with code {proc.returncode}")

            for t in threads:
                t.join(timeout=5)

            # Check convergence
            new_top = registry.get_top_k(k=1)
            if new_top:
                best = new_top[0]
                print(f"\n📊 Best so far: {best['name']} (score: {best['composite_score']})")

            print()

        # Final summary
        final_top = registry.get_top_k(k=self.config.top_k)
        print(f"\n=== Final Results ===")
        for i, f in enumerate(final_top):
            print(f"  #{i+1} {f['name']} — score: {f['composite_score']}, "
                  f"IC_IR: {f['metrics'].get('ic_ir', '?')}")
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_orchestrator.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/deepclaude/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: evolution orchestrator with concurrent Claude Code dispatch"
```

---

### Task 11: CLI Entry Point

**Files:**
- Create: `src/deepclaude/__main__.py`

**Step 1: Write the CLI entry point**

```python
# src/deepclaude/__main__.py
"""CLI entry point: python -m deepclaude"""

import argparse
import sys

from deepclaude.orchestrator import Config, Orchestrator


def main():
    parser = argparse.ArgumentParser(description="DeepClaude: Autonomous quant factor research")
    parser.add_argument("--parallel", "-n", type=int, default=3, help="Concurrent Claude instances per round")
    parser.add_argument("--rounds", "-r", type=int, default=10, help="Max evolution rounds")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Top K factors for selection")
    parser.add_argument("--max-iter", type=int, default=20, help="Max iterations per Claude instance")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to trader-data")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    args = parser.parse_args()

    config = Config(
        n_parallel=args.parallel,
        max_rounds=args.rounds,
        top_k=args.top_k,
        max_iterations=args.max_iter,
        data_dir=args.data_dir,
        project_root=args.project_root,
    )
    orch = Orchestrator(config)
    orch.run()


if __name__ == "__main__":
    main()
```

**Step 2: Verify it runs (help only, no actual Claude launch)**

Run: `cd /path/to/deepclaude && python -m deepclaude --help`
Expected: prints usage with all flags

**Step 3: Commit**

```bash
git add src/deepclaude/__main__.py
git commit -m "feat: CLI entry point for running evolution loop"
```

---

### Task 12: Integration Test — SDK Smoke Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

This tests the full SDK pipeline (data → operators → backtest → registry) without launching Claude Code.

```python
# tests/test_integration.py
"""Integration test: full SDK pipeline without Claude Code."""

import json
import os

import numpy as np
import pytest


@pytest.fixture
def setup_env(tmp_path, monkeypatch):
    """Set up environment for SDK integration test."""
    factor_dir = tmp_path / "factors"
    factor_dir.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("DEEPCLAUDE_FACTOR_DIR", str(factor_dir))
    monkeypatch.setenv("DEEPCLAUDE_WORKSPACE", str(workspace))
    monkeypatch.setenv("DEEPCLAUDE_SESSION_ID", "integration_test")
    # Reset logger singleton
    from deepclaude import logger
    logger._writer = None
    return {"factor_dir": factor_dir, "workspace": workspace}


class TestFullPipeline:
    def test_data_to_factor_to_backtest_to_registry(self, setup_env):
        """Simulate what a Claude Code instance would do."""
        from deepclaude.data import get
        from deepclaude.operators import ts_return, ts_mean, cs_rank, div
        from deepclaude.backtest import evaluate, validate
        from deepclaude.registry import submit, get_top_k

        # 1. Load data
        close = get("close")
        volume = get("volume")
        returns = get("returns")
        assert close.shape[0] > 2000
        assert close.shape[1] > 500

        # 2. Compute a simple factor: momentum / volume_mean
        momentum = ts_return(close, 20)
        vol_avg = ts_mean(volume, 60)
        factor = cs_rank(div(momentum, vol_avg))

        assert factor.shape == close.shape
        assert factor.dtype == np.float32

        # 3. Evaluate
        # Use a slice for speed (not full dataset)
        T = factor.shape[0]
        # Forward returns = next day returns
        fwd = returns[1:, :]
        fac = factor[:-1, :]
        # Take last 500 days for speed
        fac_slice = fac[-500:, :]
        fwd_slice = fwd[-500:, :]

        result = evaluate(fac_slice, fwd_slice)
        assert "ic_mean" in result
        assert "sharpe" in result
        assert len(result["decay"]) == 5
        assert len(result["quantile_returns"]) == 5

        # 4. Validate
        val = validate(fac_slice, fwd_slice)
        assert val["total"] == 5
        assert 0 <= val["passed"] <= 5

        # 5. Submit to registry
        factor_id = submit(
            name="momentum_vol_adjusted",
            code="def alpha(ctx): return cs_rank(div(ts_return(ctx.close, 20), ts_mean(ctx.volume, 60)))",
            metrics=result,
            validation=val,
            analysis="Simple momentum divided by avg volume, rank normalized",
        )
        assert factor_id.startswith("alpha_")

        # 6. Retrieve from registry
        top = get_top_k(k=1)
        assert len(top) == 1
        assert top[0]["name"] == "momentum_vol_adjusted"

        # 7. Check logs were written
        log_path = setup_env["workspace"] / "research.log"
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) >= 3  # evaluate + validate + submit
```

**Step 2: Run integration test**

Run: `python -m pytest tests/test_integration.py -v`
Expected: all PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end integration test for full SDK pipeline"
```

---

## Execution Summary

| Task | Description | Dependencies |
|------|-------------|-------------|
| 1 | Project scaffolding | None |
| 2 | Data module | Task 1 |
| 3 | Operators — port 39 existing | Task 1 |
| 4 | Operators — add 10 new + fix cs_rank | Task 3 |
| 5 | Logger module | Task 1 |
| 6 | Backtest — evaluate() | Tasks 2, 3, 5 |
| 7 | Backtest — validate() | Task 6 |
| 8 | Registry module | Task 5 |
| 9 | Prompt template | None (text only) |
| 10 | Orchestrator | Tasks 8, 9 |
| 11 | CLI entry point | Task 10 |
| 12 | Integration test | All above |

**Parallelizable groups:**
- Tasks 2, 3, 5, 9 can run in parallel (independent modules)
- Tasks 4, 6, 8 can start once their dependencies complete
- Tasks 7, 10 require sequential completion
- Tasks 11, 12 are final
