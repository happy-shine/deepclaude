---
name: deepclaude-sdk
description: DeepClaude SDK API reference — data loading, operators, backtest evaluation, validation gates, and factor registry. Use when writing factor code or calling SDK functions.
---

# DeepClaude SDK API 文档

## data — 数据加载

```python
from deepclaude.data import get, get_dates, get_symbols, get_universe, get_benchmark, get_universe_mask

close = get("close")                           # (T, N) float32
high  = get("high", start="2020-01-01")        # 带时间过滤
dates = get_dates()                            # datetime64[ns] array
symbols = get_symbols()                        # list[str], 610只
universe = get_universe("spx", month="2025-01") # 该月SPX成分股
benchmark = get_benchmark("qqq")               # 1D float32
spx_mask = get_universe_mask("spx")            # (T, N) bool, 按月SPX成分
```

字段：open, high, low, close, volume, returns。全部 LRU 缓存，并发安全。

## operators — 49 个 Numba JIT 算子

所有算子输入输出均为 `(T, N) float32`，可任意组合嵌套。

### 时序算子（24个）
```
ts_return(data, window)          # 窗口收益率
ts_mean(data, window)            # 滚动均值
ts_std(data, window)             # 滚动标准差
ts_max(data, window)             # 滚动最大值
ts_min(data, window)             # 滚动最小值
ts_rank(data, window)            # 时序百分位排名 [0,1]
ts_slope(data, window)           # 线性回归斜率
ts_r2(data, window)              # 线性趋势R²
ts_hurst(data, window)           # Hurst指数（趋势/均值回归）
ts_skew(data, window)            # 滚动偏度
ts_kurt(data, window)            # 滚动峰度
ts_ema(data, span)               # 指数移动平均
ts_corr(data1, data2, window)    # 滚动相关系数
ts_pct_positive(data, window)    # 正值占比
ts_high_dist(data, window)       # 距N日高点距离
ts_argmax(data, window)          # 最大值位置 [0,1]
ts_argmin(data, window)          # 最小值位置 [0,1]
ts_sum_if(data, cond, window)    # 条件求和
ts_count_if(cond, window)        # 条件计数
ts_delay(data, period)           # 滞后值
ts_delta(data, period)           # 一阶差分
ts_sum(data, window)             # 滚动求和
ts_decay_linear(data, window)    # 线性衰减加权均值
ts_covariance(d1, d2, window)    # 滚动协方差
ts_autocorr(data, window, lag)   # 滚动自相关
ts_regression_residual(y, x, w)  # 回归残差
ts_product(data, window)         # 滚动乘积
ts_quantile(data, window, q)     # 滚动分位数
```

### 截面算子（5个）
```
cs_rank(data)                    # 截面百分位排名 [0,1]
cs_zscore(data)                  # 截面Z分数
cs_group_rank(data, groups)      # 组内排名
cs_demean(data, groups)          # 组去均值
cs_percentile(data, q)           # 截面分位数（广播）
```

### 算术（9个）+ 逻辑（7个）
```
add, sub, mul, div, neg, abs_op, log_op, sign, pow_op
gt, lt, if_op, and_op, or_op, max_op, min_op
```

优先使用预编译算子。如果无法表达你的想法，可用 numpy 向量化，**禁止 for 循环遍历股票**。

## backtest — 回测评估

```python
from deepclaude.backtest import evaluate, validate

result = evaluate(factor, forward_returns, universe_mask=spx_mask)
# 返回 dict:
#   ic_mean, ic_ir, ic_positive_pct     — IC相关
#   long_return, long_sharpe, max_drawdown — 多头收益（top 30等权，含10bps成本）
#   turnover, cost_drag                  — 换手与成本
#   monotonicity                         — 分层单调性
#   decay: [1d,2d,3d,4d,5d]             — IC衰减
#   ic_series, quantile_returns          — 序列数据
#   long_return_series                   — 逐日多头收益（画收益曲线用）
#   WARNING (可选)                       — 结果不可信时出现

validation = validate(factor, forward_returns, universe_mask=spx_mask)
# 5道门: param_robust(IC稳定性), time_stable, cap_neutral, beat_random, decay_slow
# passed: 通过数, total: 5, details: 各门详情
```

## registry — 因子注册

```python
from deepclaude.registry import submit, get_top_k, get_lineage

factor_id = submit(
    name="因子名称",
    code="def alpha(ctx): ...",  # 完整可执行代码
    metrics=result,
    validation=validation,
    analysis="分析文本",
    parent="alpha_007",          # 可选
)

top_k = get_top_k(k=5)          # 按综合分排序
lineage = get_lineage("alpha_042")  # 追溯血缘
```
