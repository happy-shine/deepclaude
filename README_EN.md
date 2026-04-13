# DeepClaude

Autonomous quant factor research system driven by Claude Code.

DeepClaude launches Claude Code instances as autonomous quant researchers. Each instance designs, backtests, and iterates alpha factors using a Python SDK with 49 numba JIT operators. A multi-round evolution orchestrator selects top-K factors and injects them into the next generation's prompt.

[中文](README.md)

## Architecture

```
Orchestrator (Python)
  │
  ├── Round 1 → Claude Code instance → SDK → factors/
  ├── Round 2 → Claude Code instance → SDK → factors/
  │   (top-K from Round 1 injected into prompt)
  └── ...
```

**SDK Modules:**
- `data` — Loads daily OHLCV parquet data (610 US stocks, 2015–2026), with S&P 500 universe mask
- `operators` — 49 numba JIT-compiled operators (time-series, cross-sectional, arithmetic, logic)
- `backtest` — Evaluation engine: IC/IR, Sharpe, quantile returns, turnover, IC decay; 5-gate validation
- `registry` — Atomic factor submission with composite scoring, top-K selection, lineage tracking
- `orchestrator` — Multi-round evolution loop with auto-resume on failure

## Quick Start

### Prerequisites

- Python 3.11+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated
- Daily OHLCV data in parquet format (see [Data Format](#data-format))

### Install

```bash
git clone https://github.com/happy-shine/deepclaude.git
cd deepclaude
pip install -e .
```

### Run

```bash
# Single evolution run (10 rounds, top-5 selection)
python -m deepclaude --rounds 10 --top-k 5 --data-dir /path/to/your/data

# Resume interrupted run
python -m deepclaude --resume
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPCLAUDE_DATA_DIR` | `./data` | Path to OHLCV parquet directory |
| `DEEPCLAUDE_FACTOR_DIR` | `./factors` | Factor submission output |
| `DEEPCLAUDE_WORKSPACE` | `./workspace` | Claude session workspace |
| `DEEPCLAUDE_SESSION_ID` | `local` | Session identifier |

## Data Format

Place parquet files in `DEEPCLAUDE_DATA_DIR/vbt_ready/`:

```
data/vbt_ready/
├── open.parquet      # (T, N) float32 — dates × stocks
├── high.parquet
├── low.parquet
├── close.parquet
├── volume.parquet
├── returns.parquet   # daily returns
└── spx_mask.parquet  # S&P 500 membership (bool)
```

All files share the same DatetimeIndex (rows) and ticker columns.

## SDK Usage

Claude Code instances use the SDK autonomously, but you can also use it interactively:

```python
from deepclaude.data import get, get_universe_mask
from deepclaude.operators import *
from deepclaude.backtest import evaluate, validate
from deepclaude.registry import submit

# Load data
close = get("close")
returns = get("returns")
spx_mask = get_universe_mask("spx")

# Compute factor
factor = ts_zscore(ts_returns(close, 20), 60)

# Evaluate (align factor[:-1] with returns[1:])
result = evaluate(factor[:-1], returns[1:], universe_mask=spx_mask[:-1])
print(f"IC_IR: {result['ic_ir']:.3f}, Sharpe: {result['long_sharpe']:.3f}")

# Validate (5 anti-overfit gates)
gates = validate(factor[:-1], returns[1:], universe_mask=spx_mask[:-1])

# Submit to registry
submit("my_factor", factor, "import code...", result, gates, "Analysis notes")
```

## Operators

49 numba JIT-compiled operators across 4 categories:

**Time-series:** `ts_mean`, `ts_std`, `ts_zscore`, `ts_returns`, `ts_log_returns`, `ts_delta`, `ts_delay`, `ts_sum`, `ts_product`, `ts_min`, `ts_max`, `ts_argmin`, `ts_argmax`, `ts_rank`, `ts_skew`, `ts_kurt`, `ts_corr`, `ts_cov`, `ts_regression_residual`, `ts_linear_slope`, `ts_weighted_mean`, `ts_decay_linear`, `ts_momentum`, `ts_ema`, `ts_RSI`

**Cross-sectional:** `cs_rank`, `cs_zscore`, `cs_demean`, `cs_percentile`, `cs_winsorize`, `cs_normalize`

**Arithmetic:** `log1p`, `sign`, `abs_val`, `power`, `clip`, `diff`, `scale`, `add`, `subtract`, `multiply`, `divide`, `where`

**Logic:** `greater`, `less`, `and_op`, `or_op`, `not_op`, `if_else`

## Backtest

**Evaluation metrics:** IC mean, IC std, IC IR, positive IC ratio, quantile returns (5 bins), long return, long Sharpe, long max drawdown, monotonicity, turnover, IC decay curve

**Validation gates:**
1. IC stability (CV < 1.0)
2. Time stability (positive IC in ≥60% of years)
3. Cap neutrality (large-cap IC ≥ 50% of overall)
4. Beat random (IC > 95th percentile of 100 shuffled baselines)
5. Decay slowness (IC at lag-5 ≥ 30% of lag-1)

Built-in 10bps transaction costs and 30-stock portfolio cap.

## Project Structure

```
deepclaude/
├── src/deepclaude/
│   ├── config.py            # Environment-based configuration
│   ├── data.py              # Parquet data loading with LRU cache
│   ├── operators.py         # 49 numba JIT operators
│   ├── backtest.py          # Evaluate & validate engine
│   ├── registry.py          # Factor storage & scoring
│   ├── orchestrator.py      # Evolution loop with resume
│   ├── logger.py            # JSONL structured logging
│   ├── prompt_template.md   # Claude researcher prompt
│   └── report_template.html # HTML report template
├── tests/                   # Test suite
├── docs/plans/              # Design & implementation docs
└── pyproject.toml
```

## License

MIT
