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

        # 3. Evaluate (use last 500 days for speed)
        fwd = returns[1:, :]
        fac = factor[:-1, :]
        fac_slice = fac[-500:, :]
        fwd_slice = fwd[-500:, :]

        result = evaluate(fac_slice, fwd_slice)
        assert "ic_mean" in result
        assert "long_sharpe" in result
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
