# DeepClaude: Claude Code 驱动的自主量化因子研究系统

## 概述

DeepClaude 是一个基于 Claude Code CLI 的自主量化因子研究系统。Python 编排器并发调度多个 Claude Code 实例，每个实例自治地完成因子设计、回测、迭代优化的完整闭环。跨实例通过进化选择（收集 top K → 注入下一轮 prompt）实现因子种群的持续进化。

项目提供底层 SDK（`deepclaude`），让 Claude Code 生成的代码直接调用高性能数据读取、numba JIT 算子、回测评估等能力，避免重复造轮子。

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                    Python 编排器                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │ 进化控制器                                         │   │
│  │ - 管理因子种群（FactorRegistry）                    │   │
│  │ - 选择 top K 父代 → 构造 prompt                    │   │
│  │ - 并发启动 N 个 Claude Code 实例                    │   │
│  │ - 收集结果 → 检查收敛 → 下一轮                      │   │
│  └──────────┬───────────────────────────────────────┘   │
│             │ 并发启动                                    │
│  ┌──────────▼──────────────────────────────────────┐    │
│  │ Claude Code 实例 × N                              │    │
│  │                                                    │    │
│  │  claude --print --dangerously-skip-permissions     │    │
│  │         --output-format stream-json --verbose      │    │
│  │         --cwd {workspace}                          │    │
│  │                                                    │    │
│  │  每个实例自治执行：                                  │    │
│  │  设计因子 → 自审 → 回测 → 分析 → 迭代/提交           │    │
│  │  通过 bash 调用 deepclaude SDK                      │    │
│  └────────────────────────────────────────────────┘    │
│             │ 调用                                       │
│  ┌──────────▼──────────────────────────────────────┐    │
│  │ deepclaude SDK                                     │    │
│  │ ┌────────┐ ┌───────────┐ ┌──────────┐ ┌────────┐ │    │
│  │ │  data  │ │ operators │ │ backtest │ │registry│ │    │
│  │ └────────┘ └───────────┘ └──────────┘ └────────┘ │    │
│  │ ┌────────┐                                        │    │
│  │ │ logger │                                        │    │
│  │ └────────┘                                        │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## SDK 模块设计

### 1. `data` — 数据层

从 `/Users/shine/trader-data/` 读取预处理好的 parquet 宽表。

**数据源：**
- `vbt_ready/` 下的宽表：open, high, low, close, volume, returns
- 每个 parquet 为 `(T, N) float32`，T=2831 日，N=611 只美股，2015-2026
- `spx_history/` S&P 500 成分股历史
- `qqq_close.parquet` 基准收盘价

**接口：**
```python
from deepclaude.data import get, get_universe, get_benchmark

close = get("close")                          # np.ndarray (T, N) float32
high  = get("high", start="2020-01-01")       # 带时间过滤
universe = get_universe("spx", month="2025-01")
benchmark = get_benchmark("qqq")
```

**性能策略：**
- 内存 LRU 缓存，首次读 parquet，后续直接返回 numpy view
- 数据为只读共享，N 个 Claude 实例并发读无冲突
- float32 全程，611 只股票 × 2831 天 ≈ 每字段 6.5MB，全部字段 < 40MB

### 2. `operators` — 算子库

全部 `@numba.njit(parallel=True, cache=True)`，输入输出均为 `(T, N) float32`。

**已有算子（39个，来自 DeepGSC operators.py）：**

| 类别 | 算子 | 数量 |
|------|------|------|
| 时序 | ts_return, ts_mean, ts_std, ts_max, ts_min, ts_rank, ts_slope, ts_r2, ts_hurst, ts_pct_positive, ts_high_dist, ts_corr, ts_skew, ts_kurt, ts_argmax, ts_argmin, ts_sum_if, ts_count_if, ts_ema | 19 |
| 截面 | cs_rank, cs_zscore, cs_group_rank, cs_demean | 4 |
| 算术 | add, sub, mul, div, neg, abs_op, log_op, sign, pow_op | 9 |
| 逻辑 | gt, lt, if_op, and_op, or_op, max_op, min_op | 7 |

**需补充的算子（10个）：**

| 算子 | 用途 | 重要性 |
|------|------|--------|
| ts_delay | 滞后值 | 必须 |
| ts_delta | 一阶差分 | 必须 |
| ts_sum | 滚动求和 | 必须 |
| ts_decay_linear | 线性衰减加权均值 | 高（WQ101 高频使用） |
| ts_covariance | 协方差 | 高（beta 计算） |
| ts_autocorr | 自相关系数 | 高（动量持续性） |
| ts_regression_residual | 回归残差 | 高（行业中性化） |
| ts_product | 累积乘积 | 中 |
| ts_quantile | 滚动分位数 | 中 |
| cs_percentile | 截面百分位 | 中 |

**已有代码需修复：**
- 删除未使用的 `_nan_safe_div`
- `cs_rank` 从 O(N²) 优化为 O(N log N)（手写 merge sort）

**性能降级策略（两层）：**
1. **预编译算子**（~500ms）— 优先使用，覆盖 95%+ 因子
2. **numpy 向量化**（~2-5s）— Claude 自由编写 numpy 代码作为 fallback

禁止纯 Python for 循环遍历股票。

### 3. `backtest` — 回测评估引擎

内部全部 numba + numpy，不用任何框架。

**`evaluate()` — 12 项指标：**

```python
result = evaluate(factor_values, forward_returns, split="train")
# split: "train" (2016-2022, 默认) | "test" (2023-2026)
# 2015 为 warmup 期，不参与评估
```

| 指标 | 说明 |
|------|------|
| ic_mean | IC 均值 |
| ic_ir | IC 信息比率 = mean(IC) / std(IC) |
| long_short_return | 多空年化收益 |
| max_drawdown | 最大回撤 |
| turnover | 换手率 |
| sharpe | 多空 Sharpe 比率 |
| ic_positive_pct | IC 为正天数占比 |
| long_return | 纯多头年化收益 |
| decay | IC 衰减序列 (1d~5d) |
| monotonicity | 分层收益单调性 |
| ic_series | 逐期 IC 序列 |
| quantile_returns | 5 层分层收益 |

**仓位管理：**
- 因子返回 `(T, N) float32` → 等权分层回测
- 因子返回 `(factor_values, weights)` 元组 → 自定义权重，0 表示空仓
- `evaluate()` 自动检测返回类型

**`validate()` — 5 道反过拟合门：**

```python
validation = validate(factor_values, forward_returns)
```

| 门 | 标准 | 说明 |
|---|------|------|
| 参数扰动 | 窗口±10%后 IC 变化 < 30% | 参数敏感 = 过拟合 |
| 时间稳定 | 5 段中 ≥ 4 段 IC 为正 | 不能只在某段时间有效 |
| 跨市值 | 大/中/小盘都正 IC | 不能只在某个市值段有效 |
| 随机基线 | 优于 95% 的随机因子 | 排除运气成分 |
| 因子衰减 | 5 天后 IC 仍 > 50% 初始值 | 信号要有持续性 |

通过 3 道以上值得提交，全部通过是理想情况。

**数据分割：**
- 启动期 2015：算子 warmup（如 MA200），不参与评估
- 训练集 2016-2022：因子设计和回测迭代
- 测试集 2023-2026：submit 前一次性验证，禁止反复调参

### 4. `registry` — 因子注册表

**接口：**

```python
from deepclaude.registry import submit, get_top_k, get_lineage

submit(
    name="momentum_vol_adjusted",
    code="def alpha(ctx): ...",
    metrics=result,
    validation=validation,
    analysis="动量除以波动率，逻辑是...",
    parent="alpha_007",
)

top_k = get_top_k(k=5, sort_by="composite_score")
lineage = get_lineage("momentum_vol_adjusted")
```

**存储：** `{DEEPCLAUDE_FACTOR_DIR}/{factor_id}.json`

```json
{
    "id": "alpha_042",
    "name": "momentum_vol_adjusted",
    "session_id": "r001_i003",
    "code": "def alpha(ctx): ...",
    "metrics": {"ic_mean": 0.032, "ic_ir": 0.45, ...},
    "validation": {"passed": 4, "total": 5, ...},
    "composite_score": 0.72,
    "analysis": "...",
    "parent": "alpha_007",
    "created_at": "2026-04-08T14:23:03"
}
```

**排序：** 加权综合分（权重可配置，默认 ic_ir 为主）。

**去重：** 由 Claude 基于先验知识判断，SDK 不存因子值矩阵。

**并发安全：** 原子写入（先写 `tmp_{uuid}.json` 再 `os.rename`）。

### 5. `logger` — 日志模块

**两层日志：**

| 层 | 来源 | 格式 | 位置 |
|---|------|------|------|
| Claude Code 进程流 | `--output-format stream-json` | JSON events | 编排器捕获并解析 |
| SDK 调用日志 | evaluate/validate/submit 自动记录 | JSONL | `{workspace}/research.log` |

SDK 日志示例：
```jsonl
{"ts": "2026-04-08T14:23:01", "session": "r001_i003", "event": "evaluate", "split": "train", "ic_mean": 0.032, "ic_ir": 0.45}
{"ts": "2026-04-08T14:23:02", "session": "r001_i003", "event": "validate", "passed": 4, "details": {...}}
{"ts": "2026-04-08T14:23:03", "session": "r001_i003", "event": "submit", "name": "momentum_vol_adj", "parent": "alpha_007"}
```

对 Claude 透明，无需在 prompt 中提及。

## 编排器设计

### 进化控制器

```python
class Orchestrator:
    def run(self, config: Config):
        for round in range(config.max_rounds):
            top_k = registry.get_top_k(k=config.top_k)
            prompt = build_prompt(top_k, config)

            processes = []
            for i in range(config.n_parallel):
                session_id = f"r{round:03d}_i{i:03d}"
                workspace = f"{config.project_root}/workspace/{session_id}"
                p = launch_claude(prompt, workspace, session_id)
                processes.append(p)

            wait_all(processes)  # 实时解析 stream-json 输出

            if check_convergence(registry, config):
                break

        generate_final_report(registry, config)
```

### Claude Code 调用

```python
def launch_claude(prompt, workspace, session_id):
    cmd = [
        "claude",
        "--print",
        "--dangerously-skip-permissions",
        "--output-format", "stream-json",
        "--verbose",
        "--cwd", workspace,
        "-p", prompt,
    ]
    env = {
        "DEEPCLAUDE_DATA_DIR": data_dir,
        "DEEPCLAUDE_FACTOR_DIR": factor_dir,
        "DEEPCLAUDE_WORKSPACE": workspace,
        "DEEPCLAUDE_SESSION_ID": session_id,
    }
    return subprocess.Popen(cmd, env=env, stdout=PIPE, stderr=PIPE)
```

### 可配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| n_parallel | 3 | 每轮并发 Claude 实例数 |
| max_rounds | 10 | 最大进化轮次 |
| top_k | 5 | 每轮选择的优秀因子数 |
| max_iterations | 20 | 单实例最大迭代轮次 |
| composite_weights | {"ic_ir": 0.3, "sharpe": 0.2, "monotonicity": 0.15, "ic_positive_pct": 0.15, "long_return": 0.1, "decay": 0.1} | 综合分权重 |

默认深度优先：少并发（3）、多轮次（10）、单实例多迭代（20）。

## 工作区隔离

```
deepclaude/
├── trader-data/            # 共享只读 — 所有实例读同一份
│   └── vbt_ready/
├── factors/                # 共享注册表 — 原子写入
│   ├── alpha_001.json
│   └── alpha_002.json
└── workspace/              # 每实例独立
    ├── r001_i001/
    │   ├── research.log
    │   ├── report.html
    │   └── scratch/
    ├── r001_i002/
    └── r002_i001/
```

| 资源 | 隔离策略 | 原因 |
|------|---------|------|
| 数据 (parquet) | 共享只读 | 不变，并发读无冲突 |
| 注册表 (factors/) | 共享写入 + 原子操作 | 跨实例比较需要 |
| 日志/报告/临时文件 | 每实例独立目录 | 避免文件冲突 |
| session_id | 环境变量注入 | SDK 自动获取，Claude 无感 |

## Prompt 模板

完整 prompt 见 `src/prompt_template.md`，核心结构：

1. **角色定义** — 量化因子研究员
2. **重要规则** — 全程自主不暂停，最终生成中文报告
3. **SDK 接口文档** — data/operators/backtest/registry 完整 API
4. **数据分割** — warmup 2015 / train 2016-2022 / test 2023-2026
5. **反过拟合验证** — 5 道门的标准和使用方式
6. **仓位管理** — 支持空仓/减仓，两种返回值约定
7. **研究流程** — 双环结构（内环：单因子迭代；外环：多因子探索）
8. **上一轮优秀因子** — `{top_k_factors}` 编排器注入
9. **资源约束** — 最大迭代次数
10. **错误处理** — 自修复，连续 3 次失败换方向
11. **最终输出** — 中文 HTML 报告（概览、因子详情、探索轨迹、结论）

模板变量：`{output_dir}`, `{top_k_factors}`, `{max_iterations}`

## 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 数据存储 | Parquet 文件 | 列式存储，pandas 直接加载，并发读无锁 |
| 内部数据结构 | numpy ndarray float32 | 零抽象开销，内存 < 40MB，CPU cache 友好 |
| 算子加速 | numba @njit(parallel=True) | C 级性能，prange 多核并行 |
| 因子注册表 | JSON 文件 + 原子 rename | 简单可靠，无需数据库 |
| 进程调度 | subprocess.Popen | 足够，无需 Celery（不是长时任务队列） |
| Claude Code 调用 | CLI `claude --print` | 完整工具能力，自治编排 |

## 非目标

- 不做实盘交易执行
- 不做 A 股数据支持（首期）
- 不做 Web UI（CLI + HTML 报告足够）
- 不做因子值持久化（只存代码和指标，不存矩阵）
