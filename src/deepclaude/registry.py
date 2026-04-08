"""Factor registry with atomic file writes and composite scoring."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from deepclaude import logger as _logger

DEFAULT_WEIGHTS = {
    "ic_ir": 0.25,
    "long_sharpe": 0.25,
    "monotonicity": 0.15,
    "ic_positive_pct": 0.15,
    "long_return": 0.10,
    "decay": 0.10,
}


def _factor_dir() -> Path:
    d = Path(os.environ.get("DEEPCLAUDE_FACTOR_DIR", "factors"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _compute_composite_score(metrics: dict, weights: dict | None = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    score = 0.0
    for key, weight in w.items():
        val = metrics.get(key, 0.0)
        if key == "decay" and isinstance(val, list):
            val = sum(val) / len(val) if val else 0.0
        if isinstance(val, (int, float)):
            score += float(val) * weight
    return round(score, 6)


def _next_id() -> str:
    d = _factor_dir()
    existing = list(d.glob("alpha_*.json"))
    if not existing:
        return "alpha_001"
    nums = []
    for f in existing:
        try:
            nums.append(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    next_num = max(nums) + 1 if nums else 1
    return f"alpha_{next_num:03d}"


def submit(
    name: str,
    code: str,
    metrics: dict,
    validation: dict,
    analysis: str,
    parent: str | None = None,
    weights: dict | None = None,
) -> str:
    """Submit a factor to the registry. Returns factor ID."""
    factor_id = _next_id()
    composite = _compute_composite_score(metrics, weights)

    entry = {
        "id": factor_id,
        "name": name,
        "session_id": os.environ.get("DEEPCLAUDE_SESSION_ID", "local"),
        "code": code,
        "metrics": metrics,
        "validation": validation,
        "composite_score": composite,
        "analysis": analysis,
        "parent": parent,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    d = _factor_dir()
    tmp_path = d / f"tmp_{uuid.uuid4().hex}.json"
    final_path = d / f"{factor_id}.json"
    tmp_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
    os.rename(str(tmp_path), str(final_path))

    _logger.log("submit", factor_id=factor_id, name=name, composite_score=composite, parent=parent)

    return factor_id


def _load_all() -> list[dict]:
    d = _factor_dir()
    entries = []
    for f in d.glob("alpha_*.json"):
        try:
            entries.append(json.loads(f.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return entries


def get_top_k(k: int = 5, sort_by: str = "composite_score") -> list[dict]:
    """Return top K factors sorted by composite score descending."""
    entries = _load_all()
    entries.sort(key=lambda e: e.get(sort_by, 0.0), reverse=True)
    return entries[:k]


def get_lineage(factor_id: str) -> list[dict]:
    """Trace lineage from factor back to root."""
    all_entries = {e["id"]: e for e in _load_all()}
    chain = []
    current = factor_id
    seen = set()
    while current and current in all_entries and current not in seen:
        seen.add(current)
        entry = all_entries[current]
        chain.append(entry)
        current = entry.get("parent")
    return chain
