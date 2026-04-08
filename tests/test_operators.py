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
        data = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float32)
        result = ops.ts_decay_linear(data, 3)
        expected = (10 * 1 + 20 * 2 + 30 * 3) / 6.0
        np.testing.assert_allclose(result[2, 0], expected, atol=1e-4)


class TestTsCovariance:
    def test_shape_dtype(self, sample_data):
        result = ops.ts_covariance(sample_data, sample_data, 20)
        assert result.shape == sample_data.shape
        assert result.dtype == np.float32


class TestTsAutocorr:
    def test_shape_dtype(self, sample_data):
        result = ops.ts_autocorr(sample_data, 20, 1)
        assert result.shape == sample_data.shape
        assert result.dtype == np.float32


class TestTsRegressionResidual:
    def test_shape_dtype(self, sample_data):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(sample_data.shape).astype(np.float32)
        result = ops.ts_regression_residual(sample_data, x, 20)
        assert result.shape == sample_data.shape
        assert result.dtype == np.float32


class TestTsProduct:
    def test_basic(self):
        data = np.full((5, 2), 2.0, dtype=np.float32)
        result = ops.ts_product(data, 3)
        np.testing.assert_allclose(result[2, :], 8.0, atol=1e-4)


class TestTsQuantile:
    def test_shape_dtype(self, sample_data):
        result = ops.ts_quantile(sample_data, 20, 0.5)
        assert result.shape == sample_data.shape
        assert result.dtype == np.float32


class TestCsPercentile:
    def test_shape_dtype(self, sample_data):
        result = ops.cs_percentile(sample_data, 0.5)
        assert result.shape == sample_data.shape
        assert result.dtype == np.float32


class TestCsRankOptimized:
    def test_range(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 100)).astype(np.float32)
        result = ops.cs_rank(data)
        valid = result[~np.isnan(result)]
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0

    def test_nan_handling(self):
        data = np.array([[1.0, np.nan, 3.0, 2.0]], dtype=np.float32)
        result = ops.cs_rank(data)
        assert np.isnan(result[0, 1])
        np.testing.assert_allclose(result[0, 0], 0.0, atol=1e-5)
        np.testing.assert_allclose(result[0, 3], 1 / 3, atol=1e-5)
        np.testing.assert_allclose(result[0, 2], 2 / 3, atol=1e-5)
