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
