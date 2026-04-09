"""Tests for orchestrator."""

import json
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

from deepclaude.orchestrator import Config, build_prompt, Orchestrator, STATE_FILE


class TestConfig:
    def test_defaults(self):
        c = Config()
        assert c.max_rounds == 10
        assert c.top_k == 5
        assert c.max_iterations == 20

    def test_custom(self):
        c = Config(max_rounds=3, top_k=10)
        assert c.max_rounds == 3
        assert c.top_k == 10


class TestBuildPrompt:
    def test_no_top_k(self):
        prompt = build_prompt(top_k_factors=[], config=Config())
        # {output_dir} stays as placeholder — filled later per-instance in _launch_claude
        assert "{max_iterations}" not in prompt
        assert "20" in prompt

    def test_with_top_k(self):
        factors = [{"name": "test", "code": "def alpha(): ...", "metrics": {"ic_ir": 0.5}, "composite_score": 0.3, "analysis": "good"}]
        prompt = build_prompt(top_k_factors=factors, config=Config())
        assert "test" in prompt
        assert "IC_IR" in prompt or "ic_ir" in prompt.lower()


class TestOrchestrator:
    def test_workspace_creation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DEEPCLAUDE_FACTOR_DIR", str(tmp_path / "factors"))
        config = Config(project_root=str(tmp_path))
        orch = Orchestrator(config)
        ws = orch._make_workspace("r001")
        assert Path(ws).exists()
        assert (Path(ws) / "scratch").exists()

    def test_state_save_load(self, tmp_path):
        config = Config(project_root=str(tmp_path))
        orch = Orchestrator(config)

        orch._save_state("20260409_120000", 2, "20260409_120000_r002",
                         "claude-uuid-123", "completed")

        state = orch._load_state()
        assert state["run_ts"] == "20260409_120000"
        assert state["completed_rounds"] == 2
        assert state["current_session"]["claude_session_id"] == "claude-uuid-123"
        assert state["current_session"]["status"] == "completed"

    def test_state_load_no_file(self, tmp_path):
        config = Config(project_root=str(tmp_path))
        orch = Orchestrator(config)
        assert orch._load_state() is None

    def test_resume_no_state(self, tmp_path, capsys):
        config = Config(project_root=str(tmp_path))
        orch = Orchestrator(config)
        orch.run(resume=True)
        captured = capsys.readouterr()
        assert "No previous run to resume" in captured.out
