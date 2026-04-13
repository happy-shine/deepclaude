# DeepClaude

Claude Code 驱动的自主量化因子研究系统。

启动 Claude Code 实例作为自主量化研究员，自动设计、回测、迭代 alpha 因子。SDK 提供 49 个 numba JIT 算子、回测引擎和因子注册表。多轮进化编排器通过 top-K 选择将优秀因子注入下一代的 prompt。

[English](README_EN.md)

## 架构

```
编排器 (Python)
  │
  ├── 第 1 轮 → Claude Code 实例 → SDK → factors/
  ├── 第 2 轮 → Claude Code 实例 → SDK → factors/
  │   (第 1 轮的 top-K 因子注入 prompt)
  └── ...
```

**SDK 模块：**
- `data` — 加载日频 OHLCV parquet 数据（610 只美股，2015–2026），含 S&P 500 成分股遮罩
- `operators` — 49 个 numba JIT 编译算子（时序、截面、算术、逻辑）
- `backtest` — 评估引擎：IC/IR、Sharpe、分位收益、换手率、IC 衰减；5 道验证关卡
- `registry` — 因子原子提交、复合评分、top-K 筛选、血缘追踪
- `orchestrator` — 多轮进化循环，支持断点续跑

## 快速开始

### 前置条件

- Python 3.11+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) 已安装并认证
- 日频 OHLCV 数据（parquet 格式，见[数据格式](#数据格式)）

### 安装

```bash
git clone https://github.com/happy-shine/deepclaude.git
cd deepclaude
pip install -e .
```

### 运行

```bash
# 进化运行（10 轮，top-5 选择）
python -m deepclaude --rounds 10 --top-k 5 --data-dir /path/to/your/data

# 断点续跑
python -m deepclaude --resume
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEPCLAUDE_DATA_DIR` | `./data` | OHLCV parquet 数据目录 |
| `DEEPCLAUDE_FACTOR_DIR` | `./factors` | 因子提交输出目录 |
| `DEEPCLAUDE_WORKSPACE` | `./workspace` | Claude 会话工作区 |
| `DEEPCLAUDE_SESSION_ID` | `local` | 会话标识 |

## 数据格式

将 parquet 文件放在 `DEEPCLAUDE_DATA_DIR/vbt_ready/` 下：

```
data/vbt_ready/
├── open.parquet      # (T, N) float32 — 日期 × 股票
├── high.parquet
├── low.parquet
├── close.parquet
├── volume.parquet
├── returns.parquet   # 日收益率
└── spx_mask.parquet  # S&P 500 成分股标记 (bool)
```

所有文件共享相同的 DatetimeIndex（行）和 ticker 列。

## SDK 使用

Claude Code 实例自主使用 SDK，你也可以交互式调用：

```python
from deepclaude.data import get, get_universe_mask
from deepclaude.operators import *
from deepclaude.backtest import evaluate, validate
from deepclaude.registry import submit

# 加载数据
close = get("close")
returns = get("returns")
spx_mask = get_universe_mask("spx")

# 计算因子
factor = ts_zscore(ts_returns(close, 20), 60)

# 评估（对齐：factor[:-1] 对应 returns[1:]）
result = evaluate(factor[:-1], returns[1:], universe_mask=spx_mask[:-1])
print(f"IC_IR: {result['ic_ir']:.3f}, Sharpe: {result['long_sharpe']:.3f}")

# 验证（5 道反过拟合关卡）
gates = validate(factor[:-1], returns[1:], universe_mask=spx_mask[:-1])

# 提交到注册表
submit("my_factor", factor, "import code...", result, gates, "分析笔记")
```

## 算子

49 个 numba JIT 编译算子，分 4 类：

**时序：** `ts_mean`, `ts_std`, `ts_zscore`, `ts_returns`, `ts_log_returns`, `ts_delta`, `ts_delay`, `ts_sum`, `ts_product`, `ts_min`, `ts_max`, `ts_argmin`, `ts_argmax`, `ts_rank`, `ts_skew`, `ts_kurt`, `ts_corr`, `ts_cov`, `ts_regression_residual`, `ts_linear_slope`, `ts_weighted_mean`, `ts_decay_linear`, `ts_momentum`, `ts_ema`, `ts_RSI`

**截面：** `cs_rank`, `cs_zscore`, `cs_demean`, `cs_percentile`, `cs_winsorize`, `cs_normalize`

**算术：** `log1p`, `sign`, `abs_val`, `power`, `clip`, `diff`, `scale`, `add`, `subtract`, `multiply`, `divide`, `where`

**逻辑：** `greater`, `less`, `and_op`, `or_op`, `not_op`, `if_else`

## 回测

**评估指标：** IC 均值、IC 标准差、IC IR、正 IC 比例、分位收益（5 档）、多头收益、多头 Sharpe、多头最大回撤、单调性、换手率、IC 衰减曲线

**验证关卡：**
1. IC 稳定性（CV < 1.0）
2. 时间稳定性（≥60% 年份 IC 为正）
3. 市值中性（大盘股 IC ≥ 整体的 50%）
4. 击败随机（IC > 100 次 shuffle 基线的 95 分位）
5. 衰减缓慢（lag-5 IC ≥ lag-1 的 30%）

内置 10bps 交易成本，持仓上限 30 只。

## 项目结构

```
deepclaude/
├── src/deepclaude/
│   ├── config.py            # 环境变量配置
│   ├── data.py              # Parquet 数据加载（LRU 缓存）
│   ├── operators.py         # 49 个 numba JIT 算子
│   ├── backtest.py          # 评估 & 验证引擎
│   ├── registry.py          # 因子存储 & 评分
│   ├── orchestrator.py      # 进化循环（断点续跑）
│   ├── logger.py            # JSONL 结构化日志
│   ├── prompt_template.md   # Claude 研究员 prompt
│   └── report_template.html # HTML 报告模板
├── tests/                   # 测试
├── docs/plans/              # 设计 & 实现文档
└── pyproject.toml
```

## License

MIT
