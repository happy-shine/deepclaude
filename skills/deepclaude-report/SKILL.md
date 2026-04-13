---
name: deepclaude-report
description: Generate a standardized Chinese HTML research report for DeepClaude factor research sessions. Triggers when factor research is complete and report.html needs to be generated.
---

# DeepClaude 研究报告生成

## 触发条件
因子研究结束，需要生成 `{output_dir}/report.html`。

## 数据准备

在生成报告前，你必须先准备一个 `report_data.json` 文件，保存到 `{output_dir}/report_data.json`。

### report_data.json 结构

```python
import json
import numpy as np
from deepclaude.data import get, get_dates, get_benchmark
from deepclaude.backtest import evaluate

# 为每个提交的因子收集数据
report_data = {
    "session_id": session_id,
    "date": "2026-04-08",
    "n_backtests": n_evaluate_calls,    # 回测次数（从research.log统计）
    "n_validates": n_validate_calls,     # 验证次数
    "n_submits": n_submit_calls,         # 提交次数
    "dates": [str(d)[:10] for d in get_dates()[1:]],  # 日期轴（用于收益曲线）
    "qqq_cumulative": [],  # QQQ累积收益序列（归一化到1.0起点）
    "factors": {
        "factor_key": {
            "alpha_id": "alpha_001",
            "name": "因子名称",
            "composite_score": 0.598,
            "code": "def alpha(ctx): ...",
            "analysis": "因子分析文字",
            "parent": null,
            "train_metrics": { ... },  # evaluate()返回的完整dict
            "test_metrics": { ... },   # 不同时期的evaluate()结果
            "train_ic_series": [],     # 训练期逐日IC
            "test_ic_series": [],      # 测试期逐日IC
            "long_return_series": [],  # 全时段逐日多头收益（用于收益曲线）
            "validation": { ... },     # validate()返回
            "validation_details": { ... }
        }
    }
}
```

### QQQ 基准曲线计算

```python
qqq = get_benchmark("qqq")
dates = get_dates()
# 对齐到因子日期（去掉第一天因为forward returns shift了1天）
# 归一化
qqq_cum = qqq[1:] / qqq[1]
report_data["qqq_cumulative"] = [round(float(x), 4) for x in qqq_cum]
```

### 因子收益曲线计算

```python
# 从 long_return_series 计算累积收益
daily_returns = np.array(result["long_return_series"])
cumulative = np.cumprod(1 + daily_returns).tolist()
```

## 报告 HTML 结构

参考模板：`src/deepclaude/report_template.html`

使用 ECharts CDN：`https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js`

### 整体风格

```css
背景: #0a0e17
卡片: #111827, border: #1e293b
强调色: #3b82f6 (蓝)
正值: #10b981 (绿)
负值: #ef4444 (红)
金色: #f59e0b
字体: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif
全中文
```

### Section 1: 研究概览

5 个 stat 卡片横排：回测次数、验证次数、提交因子数、最高多头Sharpe、最高综合分

### Section 2: 策略收益曲线 vs QQQ

**最重要的图表，放在排行榜之前。**

- ECharts 折线图，500px 高度
- 每个提交因子一条线 + QQQ 基准线（灰色虚线）
- Y 轴：线性刻度（`type: 'value'`），`min: 0`，格式 `Nx`
- 必须包含 `dataZoom`：底部滑块 + 鼠标滚轮缩放
- 数据：从 `long_return_series` 计算 `np.cumprod(1 + series)`
- 降采样：每 3 个点取 1 个（避免 HTML 过大）
- 颜色方案：每因子不同颜色，QQQ 灰色虚线

```javascript
dataZoom: [
    { type: 'slider', xAxisIndex: 0, start: 0, end: 100, height: 25, bottom: 5 },
    { type: 'inside', xAxisIndex: 0, start: 0, end: 100 }
],
```

### Section 3: 因子排行榜

全宽表格，按 composite_score 降序：

| 排名 | ID | 名称 | 综合分 | 训练Sharpe | 测试Sharpe | 训练IC | 测试IC | 多头收益(训) | 多头收益(测) | 单调性(训/测) | 换手率 | 验证 |

- Top 1 行金色左边框
- 正值绿色，负值红色
- 收益率显示为百分比（如 "29.6%"）

### Section 4: 因子详情（每因子一个卡片）

按 composite_score 降序循环渲染。每个卡片包含：

#### 4A: 头部
排名徽章 + alpha_id + 因子名称 + 综合分标签

#### 4B: 描述 + 代码（2列）
- 左：analysis 文字
- 右：`<pre>` 代码块，深色背景

#### 4C: 指标表 + 验证（2列）
- 左：9行指标对比表（训练 vs 测试）
  - IC均值、IC_IR、多头年化收益(%)、多头Sharpe、最大回撤(%)、换手率、IC正占比(%)、分层单调性、IC衰减比
- 右：5道验证门徽章（✓绿 / ✗红）+ 详细数值

#### 4D: 3张图表并排
1. **IC时序折线**：train_ic_series + test_ic_series，20日滚动均值平滑，训练蓝色/测试橙色，分界竖虚线
2. **分层收益柱状图**：quantile_returns Q1-Q5，训练vs测试分组柱状，Y轴转bps（×10000）
3. **IC衰减折线**：decay 1d-5d，训练vs测试

#### 4E: 2张图表并排
4. **时间稳定IC**：validation_details.time_stable_seg_ics 5段柱状，绿正红负
5. **市值分层IC**：validation_details.cap_neutral_ics 3组柱状，绿正红负

### Section 5: 探索漏斗

顶部：视觉漏斗 `[N回测] → [M验证] → [K提交]`

底部：ECharts 散点图
- 从 research.log 解析每次 evaluate 的 long_sharpe 值
- X轴 = 序号（1-N），Y轴 = long_sharpe
- submit 事件用红色菱形标记

### Section 6: 页脚

session_id、日期、数据描述

## 图表通用配置

```javascript
// 20日滚动均值（用于IC时序图平滑）
function rollingMean(arr, w) {
    var r = [];
    for (var i = 0; i < arr.length; i++) {
        if (i < w - 1) { r.push(null); continue; }
        var s = 0;
        for (var j = i - w + 1; j <= i; j++) s += arr[j];
        r.push(s / w);
    }
    return r;
}

// 所有图表必须响应窗口缩放
window.addEventListener('resize', function() { chart.resize(); });
```

## 格式化约定

| 类型 | 格式 | 示例 |
|------|------|------|
| IC | 小数点后6位 | 0.009554 |
| IC_IR | 小数点后4位 | 0.0649 |
| Sharpe | 小数点后2位 | 1.60 |
| 收益率 | 百分比1位 | 29.6% |
| 回撤 | 百分比1位 | -18.9% |
| 换手率 | 小数点后3位 | 0.288 |
| 单调性 | 小数点后2位 | 1.00 |
| 综合分 | 小数点后4位 | 0.5985 |

## 关键要求

1. HTML 完全自包含（inline CSS + inline JS），仅外部依赖 ECharts CDN
2. 全中文界面
3. 暗色主题
4. 所有数值从 report_data.json 读取，**不要硬编码**
5. 收益曲线使用 `long_return_series` 的真实逐日数据，`np.cumprod(1+series)` 计算累积收益
6. 收益曲线数据降采样（每3点取1）避免HTML过大
7. 图表必须支持 dataZoom（至少收益曲线图必须有）
8. 每个 ECharts 实例必须注册 resize 监听
