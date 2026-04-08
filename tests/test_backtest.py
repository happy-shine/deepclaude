"""Tests for backtest evaluate and validate."""

import numpy as np
import pytest

from deepclaude import backtest


@pytest.fixture
def factor_and_returns():
    """Synthetic factor with known positive IC."""
    rng = np.random.default_rng(42)
    T, N = 500, 50
    factor = rng.standard_normal((T, N)).astype(np.float32)
    noise = rng.standard_normal((T, N)).astype(np.float32) * 5
    forward_returns = (factor * 0.1 + noise).astype(np.float32)
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
