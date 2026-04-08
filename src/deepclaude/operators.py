"""Numba JIT-compiled operators for the DeepClaude factor engine.

All operators work on 2D arrays of shape (T, N) where T=dates, N=stocks.
Data is float32 throughout. NaN is handled explicitly: skipped in
aggregations, propagated in arithmetic.

Operator categories
-------------------
- Time-series  : rolling window computations along the T axis per stock
- Cross-sectional : computations across the N axis per date
- Arithmetic   : element-wise math
- Logic        : element-wise comparisons / conditionals
"""

from __future__ import annotations

import math

import numba
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_F32_NAN = np.float32(np.nan)


# ---------------------------------------------------------------------------
# Time-series operators
# ---------------------------------------------------------------------------


@numba.njit(parallel=True, cache=True)
def ts_return(data: np.ndarray, window: int) -> np.ndarray:
    """Window return rate: (x[t] - x[t-window]) / x[t-window]."""
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(window, T):
            prev = data[t - window, j]
            if math.isnan(prev) or prev == 0.0:
                out[t, j] = _F32_NAN
            elif math.isnan(data[t, j]):
                out[t, j] = _F32_NAN
            else:
                out[t, j] = (data[t, j] - prev) / prev
    return out


@numba.njit(parallel=True, cache=True)
def ts_mean(data: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean over *window* days."""
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(window - 1, T):
            s = np.float64(0.0)
            cnt = 0
            for k in range(t - window + 1, t + 1):
                v = data[k, j]
                if not math.isnan(v):
                    s += v
                    cnt += 1
            if cnt > 0:
                out[t, j] = np.float32(s / cnt)
    return out


@numba.njit(parallel=True, cache=True)
def ts_std(data: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation (sample) over *window* days."""
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(window - 1, T):
            s = np.float64(0.0)
            s2 = np.float64(0.0)
            cnt = 0
            for k in range(t - window + 1, t + 1):
                v = data[k, j]
                if not math.isnan(v):
                    s += v
                    s2 += v * v
                    cnt += 1
            if cnt > 1:
                mean = s / cnt
                var = (s2 - cnt * mean * mean) / (cnt - 1)
                if var < 0.0:
                    var = 0.0
                out[t, j] = np.float32(math.sqrt(var))
    return out


@numba.njit(parallel=True, cache=True)
def ts_max(data: np.ndarray, window: int) -> np.ndarray:
    """Rolling max over *window* days."""
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(window - 1, T):
            mx = -np.inf
            found = False
            for k in range(t - window + 1, t + 1):
                v = data[k, j]
                if not math.isnan(v):
                    if v > mx:
                        mx = v
                    found = True
            if found:
                out[t, j] = np.float32(mx)
    return out


@numba.njit(parallel=True, cache=True)
def ts_min(data: np.ndarray, window: int) -> np.ndarray:
    """Rolling min over *window* days."""
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(window - 1, T):
            mn = np.inf
            found = False
            for k in range(t - window + 1, t + 1):
                v = data[k, j]
                if not math.isnan(v):
                    if v < mn:
                        mn = v
                    found = True
            if found:
                out[t, j] = np.float32(mn)
    return out


@numba.njit(parallel=True, cache=True)
def ts_rank(data: np.ndarray, window: int) -> np.ndarray:
    """Time-series percentile rank of current value within window [0, 1]."""
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(window - 1, T):
            cur = data[t, j]
            if math.isnan(cur):
                continue
            cnt = 0
            less = 0
            for k in range(t - window + 1, t + 1):
                v = data[k, j]
                if not math.isnan(v):
                    cnt += 1
                    if v < cur:
                        less += 1
            if cnt > 0:
                out[t, j] = np.float32(less / cnt)
    return out


@numba.njit(parallel=True, cache=True)
def ts_slope(data: np.ndarray, window: int) -> np.ndarray:
    """Linear regression slope over *window* days (x = 0..window-1)."""
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(window - 1, T):
            # OLS: slope = (sum(x*y) - n*xbar*ybar) / (sum(x^2) - n*xbar^2)
            sx = np.float64(0.0)
            sy = np.float64(0.0)
            sxy = np.float64(0.0)
            sx2 = np.float64(0.0)
            cnt = 0
            for i in range(window):
                v = data[t - window + 1 + i, j]
                if not math.isnan(v):
                    x = np.float64(i)
                    sx += x
                    sy += v
                    sxy += x * v
                    sx2 += x * x
                    cnt += 1
            if cnt > 1:
                xbar = sx / cnt
                ybar = sy / cnt
                denom = sx2 - cnt * xbar * xbar
                if abs(denom) > 1e-12:
                    out[t, j] = np.float32((sxy - cnt * xbar * ybar) / denom)
    return out


@numba.njit(parallel=True, cache=True)
def ts_r2(data: np.ndarray, window: int) -> np.ndarray:
    """R-squared of linear trend over *window* days."""
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(window - 1, T):
            sx = np.float64(0.0)
            sy = np.float64(0.0)
            sxy = np.float64(0.0)
            sx2 = np.float64(0.0)
            sy2 = np.float64(0.0)
            cnt = 0
            for i in range(window):
                v = data[t - window + 1 + i, j]
                if not math.isnan(v):
                    x = np.float64(i)
                    sx += x
                    sy += v
                    sxy += x * v
                    sx2 += x * x
                    sy2 += v * v
                    cnt += 1
            if cnt > 1:
                xbar = sx / cnt
                ybar = sy / cnt
                denom_x = sx2 - cnt * xbar * xbar
                denom_y = sy2 - cnt * ybar * ybar
                if abs(denom_x) > 1e-12 and abs(denom_y) > 1e-12:
                    r = (sxy - cnt * xbar * ybar) / math.sqrt(denom_x * denom_y)
                    out[t, j] = np.float32(r * r)
                else:
                    out[t, j] = np.float32(0.0)
    return out


@numba.njit(parallel=True, cache=True)
def ts_hurst(data: np.ndarray, window: int) -> np.ndarray:
    """Simplified Hurst exponent using rescaled-range (R/S) method.

    H > 0.5 indicates trending (persistent) behaviour.
    H < 0.5 indicates mean-reverting behaviour.
    """
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(window - 1, T):
            # Collect returns in the window
            cnt = 0
            s = np.float64(0.0)
            for k in range(t - window + 1, t + 1):
                v = data[k, j]
                if not math.isnan(v):
                    s += v
                    cnt += 1
            if cnt < 4:
                continue
            mean = s / cnt
            # Compute cumulative deviations, range, and std
            cum = np.float64(0.0)
            cum_max = -np.inf
            cum_min = np.inf
            s2 = np.float64(0.0)
            for k in range(t - window + 1, t + 1):
                v = data[k, j]
                if math.isnan(v):
                    continue
                cum += (v - mean)
                if cum > cum_max:
                    cum_max = cum
                if cum < cum_min:
                    cum_min = cum
                s2 += (v - mean) * (v - mean)
            std = math.sqrt(s2 / cnt)
            if std < 1e-12:
                continue
            rs = (cum_max - cum_min) / std
            if rs > 0.0 and cnt > 1:
                out[t, j] = np.float32(math.log(rs) / math.log(cnt))
    return out


@numba.njit(parallel=True, cache=True)
def ts_pct_positive(data: np.ndarray, window: int) -> np.ndarray:
    """Percentage of positive values in window."""
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(window - 1, T):
            cnt = 0
            pos = 0
            for k in range(t - window + 1, t + 1):
                v = data[k, j]
                if not math.isnan(v):
                    cnt += 1
                    if v > 0.0:
                        pos += 1
            if cnt > 0:
                out[t, j] = np.float32(pos / cnt)
    return out


@numba.njit(parallel=True, cache=True)
def ts_high_dist(data: np.ndarray, window: int) -> np.ndarray:
    """Distance from N-day high as percentage: (high - current) / high."""
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(window - 1, T):
            cur = data[t, j]
            if math.isnan(cur):
                continue
            mx = -np.inf
            found = False
            for k in range(t - window + 1, t + 1):
                v = data[k, j]
                if not math.isnan(v):
                    if v > mx:
                        mx = v
                    found = True
            if found and mx != 0.0:
                out[t, j] = np.float32((mx - cur) / mx)
    return out


@numba.njit(parallel=True, cache=True)
def ts_corr(data1: np.ndarray, data2: np.ndarray, window: int) -> np.ndarray:
    """Rolling Pearson correlation between two (T, N) arrays."""
    T, N = data1.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(window - 1, T):
            sx = np.float64(0.0)
            sy = np.float64(0.0)
            sxy = np.float64(0.0)
            sx2 = np.float64(0.0)
            sy2 = np.float64(0.0)
            cnt = 0
            for k in range(t - window + 1, t + 1):
                x = data1[k, j]
                y = data2[k, j]
                if math.isnan(x) or math.isnan(y):
                    continue
                sx += x
                sy += y
                sxy += x * y
                sx2 += x * x
                sy2 += y * y
                cnt += 1
            if cnt > 1:
                xbar = sx / cnt
                ybar = sy / cnt
                dx = sx2 - cnt * xbar * xbar
                dy = sy2 - cnt * ybar * ybar
                if dx > 1e-12 and dy > 1e-12:
                    out[t, j] = np.float32(
                        (sxy - cnt * xbar * ybar) / math.sqrt(dx * dy)
                    )
    return out


@numba.njit(parallel=True, cache=True)
def ts_skew(data: np.ndarray, window: int) -> np.ndarray:
    """Rolling skewness over *window* days (sample-adjusted)."""
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(window - 1, T):
            s = np.float64(0.0)
            cnt = 0
            for k in range(t - window + 1, t + 1):
                v = data[k, j]
                if not math.isnan(v):
                    s += v
                    cnt += 1
            if cnt < 3:
                continue
            mean = s / cnt
            m2 = np.float64(0.0)
            m3 = np.float64(0.0)
            for k in range(t - window + 1, t + 1):
                v = data[k, j]
                if not math.isnan(v):
                    d = v - mean
                    m2 += d * d
                    m3 += d * d * d
            m2 /= cnt
            m3 /= cnt
            if m2 < 1e-12:
                out[t, j] = np.float32(0.0)
            else:
                skew_val = m3 / (m2 * math.sqrt(m2))
                # Sample adjustment: n / ((n-1)*(n-2)) * skew_val * n
                adj = (cnt * cnt) / ((cnt - 1) * (cnt - 2))
                out[t, j] = np.float32(skew_val * adj)
    return out


@numba.njit(parallel=True, cache=True)
def ts_kurt(data: np.ndarray, window: int) -> np.ndarray:
    """Rolling excess kurtosis over *window* days."""
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(window - 1, T):
            s = np.float64(0.0)
            cnt = 0
            for k in range(t - window + 1, t + 1):
                v = data[k, j]
                if not math.isnan(v):
                    s += v
                    cnt += 1
            if cnt < 4:
                continue
            mean = s / cnt
            m2 = np.float64(0.0)
            m4 = np.float64(0.0)
            for k in range(t - window + 1, t + 1):
                v = data[k, j]
                if not math.isnan(v):
                    d = v - mean
                    d2 = d * d
                    m2 += d2
                    m4 += d2 * d2
            m2 /= cnt
            m4 /= cnt
            if m2 < 1e-12:
                out[t, j] = np.float32(0.0)
            else:
                kurt_val = m4 / (m2 * m2) - 3.0
                out[t, j] = np.float32(kurt_val)
    return out


@numba.njit(parallel=True, cache=True)
def ts_argmax(data: np.ndarray, window: int) -> np.ndarray:
    """Position of max within window, normalized to [0, 1]."""
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(window - 1, T):
            mx = -np.inf
            pos = 0
            found = False
            for i in range(window):
                v = data[t - window + 1 + i, j]
                if not math.isnan(v):
                    if v > mx:
                        mx = v
                        pos = i
                    found = True
            if found and window > 1:
                out[t, j] = np.float32(pos / (window - 1))
    return out


@numba.njit(parallel=True, cache=True)
def ts_argmin(data: np.ndarray, window: int) -> np.ndarray:
    """Position of min within window, normalized to [0, 1]."""
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(window - 1, T):
            mn = np.inf
            pos = 0
            found = False
            for i in range(window):
                v = data[t - window + 1 + i, j]
                if not math.isnan(v):
                    if v < mn:
                        mn = v
                        pos = i
                    found = True
            if found and window > 1:
                out[t, j] = np.float32(pos / (window - 1))
    return out


@numba.njit(parallel=True, cache=True)
def ts_sum_if(data: np.ndarray, cond: np.ndarray, window: int) -> np.ndarray:
    """Sum of *data* where *cond* > 0 within *window*."""
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(window - 1, T):
            s = np.float64(0.0)
            cnt = 0
            for k in range(t - window + 1, t + 1):
                c = cond[k, j]
                v = data[k, j]
                if not math.isnan(c) and c > 0.0 and not math.isnan(v):
                    s += v
                    cnt += 1
            if cnt > 0:
                out[t, j] = np.float32(s)
            else:
                out[t, j] = np.float32(0.0)
    return out


@numba.njit(parallel=True, cache=True)
def ts_count_if(cond: np.ndarray, window: int) -> np.ndarray:
    """Count of values where *cond* > 0 within *window*."""
    T, N = cond.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for j in numba.prange(N):
        for t in range(window - 1, T):
            cnt = 0
            for k in range(t - window + 1, t + 1):
                c = cond[k, j]
                if not math.isnan(c) and c > 0.0:
                    cnt += 1
            out[t, j] = np.float32(cnt)
    return out


@numba.njit(parallel=True, cache=True)
def ts_ema(data: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average with given *span*.

    alpha = 2 / (span + 1).  First valid value seeds the EMA.
    """
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    alpha = np.float64(2.0 / (span + 1))
    for j in numba.prange(N):
        ema = np.float64(np.nan)
        for t in range(T):
            v = data[t, j]
            if math.isnan(v):
                if not math.isnan(ema):
                    out[t, j] = np.float32(ema)
                continue
            if math.isnan(ema):
                ema = np.float64(v)
            else:
                ema = alpha * v + (1.0 - alpha) * ema
            out[t, j] = np.float32(ema)
    return out


# ---------------------------------------------------------------------------
# Cross-sectional operators
# ---------------------------------------------------------------------------


@numba.njit(parallel=True, cache=True)
def cs_rank(data: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank per row, range [0, 1]."""
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for t in numba.prange(T):
        cnt = 0
        for j in range(N):
            if not math.isnan(data[t, j]):
                cnt += 1
        if cnt == 0:
            continue
        for j in range(N):
            v = data[t, j]
            if math.isnan(v):
                continue
            less = 0
            for k in range(N):
                u = data[t, k]
                if not math.isnan(u) and u < v:
                    less += 1
            out[t, j] = np.float32(less / cnt)
    return out


@numba.njit(parallel=True, cache=True)
def cs_zscore(data: np.ndarray) -> np.ndarray:
    """Cross-sectional Z-score per row."""
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for t in numba.prange(T):
        s = np.float64(0.0)
        s2 = np.float64(0.0)
        cnt = 0
        for j in range(N):
            v = data[t, j]
            if not math.isnan(v):
                s += v
                s2 += v * v
                cnt += 1
        if cnt < 2:
            continue
        mean = s / cnt
        var = (s2 - cnt * mean * mean) / (cnt - 1)
        if var < 1e-12:
            continue
        std = math.sqrt(var)
        for j in range(N):
            v = data[t, j]
            if not math.isnan(v):
                out[t, j] = np.float32((v - mean) / std)
    return out


@numba.njit(parallel=True, cache=True)
def cs_group_rank(data: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Within-group percentile rank.

    Parameters
    ----------
    data : (T, N) float32
    groups : (N,) int32 -- group id per stock (e.g. industry code)

    Returns
    -------
    (T, N) float32 with rank within each group in [0, 1].
    """
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for t in numba.prange(T):
        for j in range(N):
            v = data[t, j]
            if math.isnan(v):
                continue
            g = groups[j]
            cnt = 0
            less = 0
            for k in range(N):
                if groups[k] != g:
                    continue
                u = data[t, k]
                if not math.isnan(u):
                    cnt += 1
                    if u < v:
                        less += 1
            if cnt > 0:
                out[t, j] = np.float32(less / cnt)
    return out


@numba.njit(parallel=True, cache=True)
def cs_demean(data: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Group-demeaned value: x - group_mean(x).

    Parameters
    ----------
    data : (T, N) float32
    groups : (N,) int32
    """
    T, N = data.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    # Find max group id to allocate accumulators
    max_g = 0
    for j in range(N):
        if groups[j] > max_g:
            max_g = groups[j]
    n_groups = max_g + 1
    for t in numba.prange(T):
        # Compute group sums and counts
        g_sum = np.zeros(n_groups, dtype=np.float64)
        g_cnt = np.zeros(n_groups, dtype=np.int64)
        for j in range(N):
            v = data[t, j]
            if not math.isnan(v):
                g = groups[j]
                g_sum[g] += v
                g_cnt[g] += 1
        for j in range(N):
            v = data[t, j]
            if math.isnan(v):
                continue
            g = groups[j]
            if g_cnt[g] > 0:
                out[t, j] = np.float32(v - g_sum[g] / g_cnt[g])
    return out


# ---------------------------------------------------------------------------
# Arithmetic operators
# ---------------------------------------------------------------------------


@numba.njit(parallel=True, cache=True)
def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise addition."""
    T, N = a.shape
    out = np.empty((T, N), dtype=np.float32)
    for t in numba.prange(T):
        for j in range(N):
            out[t, j] = a[t, j] + b[t, j]
    return out


@numba.njit(parallel=True, cache=True)
def sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise subtraction."""
    T, N = a.shape
    out = np.empty((T, N), dtype=np.float32)
    for t in numba.prange(T):
        for j in range(N):
            out[t, j] = a[t, j] - b[t, j]
    return out


@numba.njit(parallel=True, cache=True)
def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise multiplication."""
    T, N = a.shape
    out = np.empty((T, N), dtype=np.float32)
    for t in numba.prange(T):
        for j in range(N):
            out[t, j] = a[t, j] * b[t, j]
    return out


@numba.njit(parallel=True, cache=True)
def div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise division. Divide-by-zero yields 0.0."""
    T, N = a.shape
    out = np.empty((T, N), dtype=np.float32)
    for t in numba.prange(T):
        for j in range(N):
            bv = b[t, j]
            if bv == 0.0 or math.isnan(bv):
                out[t, j] = np.float32(0.0)
            else:
                out[t, j] = a[t, j] / bv
    return out


@numba.njit(parallel=True, cache=True)
def neg(x: np.ndarray) -> np.ndarray:
    """Element-wise negation."""
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    for t in numba.prange(T):
        for j in range(N):
            out[t, j] = -x[t, j]
    return out


@numba.njit(parallel=True, cache=True)
def abs_op(x: np.ndarray) -> np.ndarray:
    """Element-wise absolute value."""
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    for t in numba.prange(T):
        for j in range(N):
            out[t, j] = np.float32(abs(x[t, j]))
    return out


@numba.njit(parallel=True, cache=True)
def log_op(x: np.ndarray) -> np.ndarray:
    """Element-wise natural logarithm. Values <= 0 yield NaN."""
    T, N = x.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for t in numba.prange(T):
        for j in range(N):
            v = x[t, j]
            if not math.isnan(v) and v > 0.0:
                out[t, j] = np.float32(math.log(v))
    return out


@numba.njit(parallel=True, cache=True)
def sign(x: np.ndarray) -> np.ndarray:
    """Element-wise sign: -1, 0, or +1 (NaN stays NaN)."""
    T, N = x.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for t in numba.prange(T):
        for j in range(N):
            v = x[t, j]
            if math.isnan(v):
                continue
            if v > 0.0:
                out[t, j] = np.float32(1.0)
            elif v < 0.0:
                out[t, j] = np.float32(-1.0)
            else:
                out[t, j] = np.float32(0.0)
    return out


@numba.njit(parallel=True, cache=True)
def pow_op(x: np.ndarray, n: float) -> np.ndarray:
    """Element-wise power: x ** n."""
    T, N = x.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for t in numba.prange(T):
        for j in range(N):
            v = x[t, j]
            if math.isnan(v):
                continue
            out[t, j] = np.float32(v ** n)
    return out


# ---------------------------------------------------------------------------
# Logic operators
# ---------------------------------------------------------------------------


@numba.njit(parallel=True, cache=True)
def gt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise greater-than: 1.0 if a > b, else 0.0."""
    T, N = a.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for t in numba.prange(T):
        for j in range(N):
            va = a[t, j]
            vb = b[t, j]
            if math.isnan(va) or math.isnan(vb):
                continue
            out[t, j] = np.float32(1.0) if va > vb else np.float32(0.0)
    return out


@numba.njit(parallel=True, cache=True)
def lt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise less-than: 1.0 if a < b, else 0.0."""
    T, N = a.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for t in numba.prange(T):
        for j in range(N):
            va = a[t, j]
            vb = b[t, j]
            if math.isnan(va) or math.isnan(vb):
                continue
            out[t, j] = np.float32(1.0) if va < vb else np.float32(0.0)
    return out


@numba.njit(parallel=True, cache=True)
def if_op(
    cond: np.ndarray, then_val: np.ndarray, else_val: np.ndarray
) -> np.ndarray:
    """Element-wise conditional: cond > 0 ? then_val : else_val."""
    T, N = cond.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for t in numba.prange(T):
        for j in range(N):
            c = cond[t, j]
            if math.isnan(c):
                continue
            if c > 0.0:
                out[t, j] = then_val[t, j]
            else:
                out[t, j] = else_val[t, j]
    return out


@numba.njit(parallel=True, cache=True)
def and_op(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise logical AND: 1.0 if both > 0, else 0.0."""
    T, N = a.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for t in numba.prange(T):
        for j in range(N):
            va = a[t, j]
            vb = b[t, j]
            if math.isnan(va) or math.isnan(vb):
                continue
            if va > 0.0 and vb > 0.0:
                out[t, j] = np.float32(1.0)
            else:
                out[t, j] = np.float32(0.0)
    return out


@numba.njit(parallel=True, cache=True)
def or_op(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise logical OR: 1.0 if either > 0, else 0.0."""
    T, N = a.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for t in numba.prange(T):
        for j in range(N):
            va = a[t, j]
            vb = b[t, j]
            if math.isnan(va) or math.isnan(vb):
                continue
            if va > 0.0 or vb > 0.0:
                out[t, j] = np.float32(1.0)
            else:
                out[t, j] = np.float32(0.0)
    return out


@numba.njit(parallel=True, cache=True)
def max_op(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise max(a, b)."""
    T, N = a.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for t in numba.prange(T):
        for j in range(N):
            va = a[t, j]
            vb = b[t, j]
            if math.isnan(va) and math.isnan(vb):
                continue
            elif math.isnan(va):
                out[t, j] = vb
            elif math.isnan(vb):
                out[t, j] = va
            else:
                out[t, j] = va if va > vb else vb
    return out


@numba.njit(parallel=True, cache=True)
def min_op(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise min(a, b)."""
    T, N = a.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for t in numba.prange(T):
        for j in range(N):
            va = a[t, j]
            vb = b[t, j]
            if math.isnan(va) and math.isnan(vb):
                continue
            elif math.isnan(va):
                out[t, j] = vb
            elif math.isnan(vb):
                out[t, j] = va
            else:
                out[t, j] = va if va < vb else vb
    return out
