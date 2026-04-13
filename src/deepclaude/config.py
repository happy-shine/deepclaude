"""Global configuration resolved from environment variables."""

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DEEPCLAUDE_DATA_DIR", "./data"))
FACTOR_DIR = Path(os.environ.get("DEEPCLAUDE_FACTOR_DIR", str(Path.cwd() / "factors")))
WORKSPACE = Path(os.environ.get("DEEPCLAUDE_WORKSPACE", str(Path.cwd() / "workspace")))
SESSION_ID = os.environ.get("DEEPCLAUDE_SESSION_ID", "local")

# 2015 is warmup only (operators like MA200 need prior data)
WARMUP_END = "2015-12-31"
# No fixed train/test split — Claude decides validation methodology
