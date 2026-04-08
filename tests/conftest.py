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
