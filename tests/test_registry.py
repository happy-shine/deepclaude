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
        files = list(tmp_registry.glob("alpha_*.json"))
        assert len(files) == 1

    def test_json_content(self, tmp_registry):
        registry.submit(
            name="test_factor",
            code="def alpha(ctx): return ctx.close",
            metrics={"ic_mean": 0.03},
            validation={"passed": 3, "total": 5},
            analysis="test",
        )
        files = list(tmp_registry.glob("alpha_*.json"))
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
        files = list(tmp_registry.glob("alpha_*.json"))
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
        files = list(tmp_registry.glob("alpha_*.json"))
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
        root_files = list(tmp_registry.glob("alpha_*.json"))
        root_id = json.loads(root_files[0].read_text())["id"]

        registry.submit(name="child", code="...", metrics={"ic_ir": 0.4},
                        validation={"passed": 4, "total": 5},
                        analysis="child", parent=root_id)
        child_files = sorted(tmp_registry.glob("alpha_*.json"))
        child_id = json.loads(child_files[-1].read_text())["id"]

        lineage = registry.get_lineage(child_id)
        assert len(lineage) == 2
        assert lineage[0]["id"] == child_id
        assert lineage[1]["id"] == root_id
