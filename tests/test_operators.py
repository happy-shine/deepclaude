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
        assert np.all(result == 0.0)


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
