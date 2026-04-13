---
name: deepclaude-backtest
description: DeepClaude backtest best practices — data alignment, transaction costs, regime analysis, and result sanity checks. Use when running backtests or analyzing results.
---

# DeepClaude 回测注意事项

## 数据对齐（最常见的 bug 源）

```python
factor_values = alpha(ctx)        # 用 t 时刻数据计算的因子
fwd = returns[1:, :]              # t+1 的收益
fac = factor_values[:-1, :]       # t 的因子值
result = evaluate(fac, fwd, universe_mask=spx_mask[:-1])
```

**factor[t] 预测 returns[t+1]**。如果用 factor[t] 对 returns[t]，就是前视偏差。

## 结果可信度

| 指标 | 正常范围 | 异常（几乎必定是bug） |
|------|---------|---------------------|
| Sharpe | 0.5 - 3.0 | > 5 |
| 年化收益 | 5% - 100% | > 200% |
| IC均值 | 0.005 - 0.05 | > 0.1 |

evaluate() 返回 `WARNING` 字段时 = 结果不可信，停下来排查。

常见原因：
- forward_returns 对齐错误（最常见）
- 因子代码访问了未来数据
- universe_mask 没对齐

## 交易成本

已内置 10bps/次换仓成本（COST_PER_TURNOVER = 0.001）。
- `cost_drag` 字段 = 年化成本拖累
- 30% 日换手 × 252天 × 0.1% ≈ 7.6% 年化成本
- 高换手因子（>30%）要确认扣费后仍有正收益

## 市场 Regime 分析

分析因子时拆分不同市场环境的表现：

```python
# 用市场月收益定义 regime
import numpy as np
mkt_ret = returns.mean(axis=1)  # 市场日均收益

# 按月聚合
# 上涨月 / 下跌月 / 震荡月 分别统计因子IC和收益
# 一个只在牛市赚钱的因子 = 低beta伪装alpha
```

在 analysis 中报告：
- 上涨月份 vs 下跌月份的 IC 和收益差异
- 高波动期 vs 低波动期的表现
- 如果因子在下跌月份 IC 为负，说明它是 beta 不是 alpha

## 仓位管理

因子可返回 `(factor_values, weights)` 元组：
- weights > 0 的股票被选入组合
- weights = 0 表示空仓
- 最多 30 只股票（硬性上限）

```python
def alpha(ctx):
    signal = cs_rank(ts_return(ctx.close, 20))
    mkt_vol = ts_std(ctx.returns.mean(axis=1, keepdims=True), 20)
    position_scale = np.where(mkt_vol > 0.03, 0.3, 1.0)
    weights = signal * position_scale
    return signal, weights
```

注意：weights 改变了选股逻辑，evaluate() 的 Sharpe 会反映 weights 后的收益。

## 反过拟合 5 道门

| 门 | 标准 | 说明 |
|---|------|------|
| IC稳定性 | IC变异系数(std/mean)<3.0 | IC波动太大=不可靠 |
| 时间稳定 | 5段中≥4段IC为正 | 不能只在某段时间有效 |
| 跨市值 | 大/中/小盘都正IC | 不能只在某个市值段有效 |
| 随机基线 | 优于95%的随机因子 | 排除运气成分 |
| 因子衰减 | 5天后IC仍>50%初始值 | 信号要有持续性 |

通过 3 道以上才值得 submit。

## 验证方法论

严禁固定二分法。推荐方法：
- Walk-Forward：滚动窗口训练→验证→前推
- 多折时序CV：2016-2026 分 N 段轮流
- 多段 holdout：不同年份组合

核心：因子必须在多个不同时间段上验证有效。
