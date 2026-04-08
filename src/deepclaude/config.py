"""Global configuration resolved from environment variables."""

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DEEPCLAUDE_DATA_DIR", "/Users/shine/trader-data"))
FACTOR_DIR = Path(os.environ.get("DEEPCLAUDE_FACTOR_DIR", str(Path.cwd() / "factors")))
WORKSPACE = Path(os.environ.get("DEEPCLAUDE_WORKSPACE", str(Path.cwd() / "workspace")))
SESSION_ID = os.environ.get("DEEPCLAUDE_SESSION_ID", "local")

# Data split boundaries
WARMUP_END = "2015-12-31"   # 2015 = warmup only
TRAIN_START = "2016-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2026-12-31"
