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
        assert T > 2000
        assert N > 500

    def test_all_fields_loadable(self):
        for field in ("open", "high", "low", "close", "volume", "returns"):
            arr = data.get(field)
            assert arr.dtype == np.float32

    def test_cache_returns_same_object(self):
        a = data.get("close")
        b = data.get("close")
        assert a is b

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
