"""Backtest evaluation engine.

All internals use numpy. No pandas, no frameworks.
"""

from __future__ import annotations

import numpy as np

from deepclaude import logger as _logger


def _rank_ic_per_row(factor: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Compute rank IC (Spearman) per row. Returns (T,) array."""
    T, N = factor.shape
    ics = np.full(T, np.nan, dtype=np.float64)
    for t in range(T):
        f_row = factor[t, :]
        r_row = returns[t, :]
        mask = ~np.isnan(f_row) & ~np.isnan(r_row)
        n = mask.sum()
        if n < 5:
            continue
        f_valid = f_row[mask]
        r_valid = r_row[mask]
        f_ranks = _rankdata(f_valid)
        r_ranks = _rankdata(r_valid)
        f_dm = f_ranks - f_ranks.mean()
        r_dm = r_ranks - r_ranks.mean()
        denom = np.sqrt((f_dm ** 2).sum() * (r_dm ** 2).sum())
        if denom < 1e-12:
            continue
        ics[t] = (f_dm * r_dm).sum() / denom
    return ics


def _rankdata(arr: np.ndarray) -> np.ndarray:
    """Simple rankdata: returns 0-based ranks."""
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(arr), dtype=np.float64)
    return ranks


def _quantile_returns(factor: np.ndarray, returns: np.ndarray, n_quantiles: int = 5) -> list[float]:
    """Mean return per quantile bucket, averaged across time."""
    T, N = factor.shape
    bucket_returns = [[] for _ in range(n_quantiles)]
    for t in range(T):
        f_row = factor[t, :]
        r_row = returns[t, :]
        mask = ~np.isnan(f_row) & ~np.isnan(r_row)
        n = mask.sum()
        if n < n_quantiles:
            continue
        f_valid = f_row[mask]
        r_valid = r_row[mask]
        ranks = _rankdata(f_valid)
        for q in range(n_quantiles):
            lo = q / n_quantiles * n
            hi = (q + 1) / n_quantiles * n
            sel = (ranks >= lo) & (ranks < hi)
            if q == n_quantiles - 1:
                sel = ranks >= lo
            if sel.sum() > 0:
                bucket_returns[q].append(float(r_valid[sel].mean()))
    return [float(np.mean(b)) if b else 0.0 for b in bucket_returns]


def _monotonicity(quantile_rets: list[float]) -> float:
    """Spearman correlation between quantile index and quantile return."""
    n = len(quantile_rets)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    y = np.array(quantile_rets, dtype=np.float64)
    x_ranks = _rankdata(x)
    y_ranks = _rankdata(y)
    xd = x_ranks - x_ranks.mean()
    yd = y_ranks - y_ranks.mean()
    denom = np.sqrt((xd ** 2).sum() * (yd ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float((xd * yd).sum() / denom)


def _long_short_returns(factor: np.ndarray, returns: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Daily long-short portfolio return series."""
    T, N = factor.shape
    ls_ret = np.zeros(T, dtype=np.float64)
    for t in range(T):
        f_row = factor[t, :]
        r_row = returns[t, :]
        mask = ~np.isnan(f_row) & ~np.isnan(r_row)
        n = mask.sum()
        if n < 10:
            continue
        f_valid = f_row[mask]
        r_valid = r_row[mask]
        if weights is not None:
            w_valid = weights[t, mask].astype(np.float64)
            w_sum = np.abs(w_valid).sum()
            if w_sum > 0:
                ls_ret[t] = (w_valid * r_valid).sum() / w_sum
        else:
            ranks = _rankdata(f_valid)
            n_valid = len(f_valid)
            top = ranks >= n_valid * 0.8
            bot = ranks < n_valid * 0.2
            if top.sum() > 0 and bot.sum() > 0:
                ls_ret[t] = r_valid[top].mean() - r_valid[bot].mean()
    return ls_ret


def _long_only_returns(factor: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Daily long-only (top quintile) portfolio return series."""
    T, N = factor.shape
    l_ret = np.zeros(T, dtype=np.float64)
    for t in range(T):
        f_row = factor[t, :]
        r_row = returns[t, :]
        mask = ~np.isnan(f_row) & ~np.isnan(r_row)
        n = mask.sum()
        if n < 10:
            continue
        f_valid = f_row[mask]
        r_valid = r_row[mask]
        ranks = _rankdata(f_valid)
        top = ranks >= len(f_valid) * 0.8
        if top.sum() > 0:
            l_ret[t] = r_valid[top].mean()
    return l_ret


def _turnover(factor: np.ndarray) -> float:
    """Average daily turnover of top quintile."""
    T, N = factor.shape
    turnovers = []
    prev_top = None
    for t in range(T):
        f_row = factor[t, :]
        mask = ~np.isnan(f_row)
        n = mask.sum()
        if n < 10:
            continue
        ranks = np.full(N, np.nan)
        f_valid = f_row[mask]
        r = _rankdata(f_valid)
        ranks[mask] = r
        top = set(np.where(ranks >= n * 0.8)[0])
        if prev_top is not None and len(top) > 0 and len(prev_top) > 0:
            overlap = len(top & prev_top)
            turnovers.append(1.0 - overlap / max(len(top), len(prev_top)))
        prev_top = top
    return float(np.mean(turnovers)) if turnovers else 0.0


def _max_drawdown(cumulative: np.ndarray) -> float:
    """Max drawdown from cumulative return series."""
    peak = np.maximum.accumulate(cumulative)
    dd = (cumulative - peak) / np.where(peak > 0, peak, 1.0)
    return float(dd.min())


def _annualize(daily_returns: np.ndarray) -> float:
    """Annualized return from daily series."""
    total = (1 + daily_returns).prod()
    n_years = len(daily_returns) / 252
    if n_years <= 0:
        return 0.0
    return float(total ** (1 / n_years) - 1)


def _sharpe(daily_returns: np.ndarray) -> float:
    """Annualized Sharpe ratio."""
    if len(daily_returns) < 10:
        return 0.0
    mean = daily_returns.mean()
    std = daily_returns.std()
    if std < 1e-12:
        return 0.0
    return float(mean / std * np.sqrt(252))


def evaluate(factor_input, forward_returns: np.ndarray, split: str = "train") -> dict:
    """Evaluate a factor against forward returns.

    Parameters
    ----------
    factor_input : np.ndarray (T, N) or tuple(factor, weights)
    forward_returns : np.ndarray (T, N)
    split : "train" or "test" (for logging only; caller passes correct slice)

    Returns
    -------
    dict with 12 evaluation metrics
    """
    weights = None
    if isinstance(factor_input, tuple):
        factor, weights = factor_input
    else:
        factor = factor_input

    ic_series_arr = _rank_ic_per_row(factor, forward_returns)
    valid_ics = ic_series_arr[~np.isnan(ic_series_arr)]

    ic_mean = float(valid_ics.mean()) if len(valid_ics) > 0 else 0.0
    ic_std = float(valid_ics.std()) if len(valid_ics) > 1 else 1.0
    ic_ir = ic_mean / ic_std if ic_std > 1e-12 else 0.0
    ic_positive_pct = float((valid_ics > 0).mean()) if len(valid_ics) > 0 else 0.0

    decay = []
    for lag in range(1, 6):
        if lag < forward_returns.shape[0]:
            shifted = np.roll(forward_returns, -lag, axis=0)
            shifted[-lag:, :] = np.nan
            ic_lag = _rank_ic_per_row(factor, shifted)
            valid = ic_lag[~np.isnan(ic_lag)]
            decay.append(float(valid.mean()) if len(valid) > 0 else 0.0)
        else:
            decay.append(0.0)

    q_rets = _quantile_returns(factor, forward_returns)
    mono = _monotonicity(q_rets)

    ls_daily = _long_short_returns(factor, forward_returns, weights)
    ls_annual = _annualize(ls_daily)
    ls_sharpe = _sharpe(ls_daily)
    cumulative = np.cumprod(1 + ls_daily)
    mdd = _max_drawdown(cumulative)

    lo_daily = _long_only_returns(factor, forward_returns)
    lo_annual = _annualize(lo_daily)

    turn = _turnover(factor)

    result = {
        "ic_mean": round(ic_mean, 6),
        "ic_ir": round(ic_ir, 4),
        "long_short_return": round(ls_annual, 4),
        "max_drawdown": round(mdd, 4),
        "turnover": round(turn, 4),
        "sharpe": round(ls_sharpe, 4),
        "ic_positive_pct": round(ic_positive_pct, 4),
        "long_return": round(lo_annual, 4),
        "decay": [round(d, 6) for d in decay],
        "monotonicity": round(mono, 4),
        "ic_series": [round(float(x), 6) for x in valid_ics.tolist()],
        "quantile_returns": [round(r, 6) for r in q_rets],
    }

    _logger.log("evaluate", split=split, **{k: v for k, v in result.items() if k not in ("ic_series", "quantile_returns")})

    return result


def validate(
    factor: np.ndarray,
    forward_returns: np.ndarray,
    n_random: int = 100,
) -> dict:
    """Run 5 anti-overfit gates on a factor.

    Returns dict with gate results, pass count, and details.
    """
    details = {}

    # Gate 1: Parameter Robustness — IC changes < 30% when factor is shifted
    base_ic = _rank_ic_per_row(factor, forward_returns)
    base_ic_mean = float(np.nanmean(base_ic))
    ic_variants = []
    for shift in [-2, -1, 1, 2]:
        shifted = np.roll(factor, shift, axis=0)
        if shift > 0:
            shifted[:shift, :] = np.nan
        else:
            shifted[shift:, :] = np.nan
        ic_v = float(np.nanmean(_rank_ic_per_row(shifted, forward_returns)))
        ic_variants.append(ic_v)
    ic_change = max(abs(v - base_ic_mean) for v in ic_variants) / max(abs(base_ic_mean), 1e-8)
    param_robust = ic_change < 0.30
    details["param_robust_max_change"] = round(ic_change, 4)

    # Gate 2: Time Stability — IC positive in >= 4 of 5 segments
    T = len(base_ic)
    seg_size = T // 5
    positive_segs = 0
    seg_ics = []
    for s in range(5):
        start = s * seg_size
        end = (s + 1) * seg_size if s < 4 else T
        seg = base_ic[start:end]
        seg_valid = seg[~np.isnan(seg)]
        seg_mean = float(seg_valid.mean()) if len(seg_valid) > 0 else 0.0
        seg_ics.append(seg_mean)
        if seg_mean > 0:
            positive_segs += 1
    time_stable = positive_segs >= 4
    details["time_stable_seg_ics"] = [round(x, 6) for x in seg_ics]
    details["time_stable_positive_count"] = positive_segs

    # Gate 3: Cap Neutral — positive IC in large/mid/small cap groups
    avg_factor = np.nanmean(np.abs(factor), axis=0)
    N = factor.shape[1]
    sorted_idx = np.argsort(avg_factor)
    third = N // 3
    cap_ics = []
    cap_names = ["small", "mid", "large"]
    cap_positive = 0
    for g, name in enumerate(cap_names):
        start = g * third
        end = (g + 1) * third if g < 2 else N
        cols = sorted_idx[start:end]
        sub_factor = factor[:, cols]
        sub_returns = forward_returns[:, cols]
        ic_g = _rank_ic_per_row(sub_factor, sub_returns)
        ic_g_mean = float(np.nanmean(ic_g))
        cap_ics.append(ic_g_mean)
        if ic_g_mean > 0:
            cap_positive += 1
    cap_neutral = cap_positive == 3
    details["cap_neutral_ics"] = {n: round(v, 6) for n, v in zip(cap_names, cap_ics)}

    # Gate 4: Beat Random — better than 95% of random factors
    random_ic_means = []
    rng = np.random.default_rng(0)
    for _ in range(n_random):
        rand_factor = rng.standard_normal(factor.shape).astype(np.float32)
        rand_ic = _rank_ic_per_row(rand_factor, forward_returns)
        random_ic_means.append(float(np.nanmean(rand_ic)))
    percentile = float(np.mean([1 if base_ic_mean > r else 0 for r in random_ic_means]))
    beat_random = percentile >= 0.95
    details["beat_random_percentile"] = round(percentile, 4)

    # Gate 5: Decay Slow — IC at day 5 still > 50% of day 1
    shifted_5 = np.roll(forward_returns, -5, axis=0)
    shifted_5[-5:, :] = np.nan
    ic_day5 = float(np.nanmean(_rank_ic_per_row(factor, shifted_5)))
    decay_ratio = ic_day5 / base_ic_mean if abs(base_ic_mean) > 1e-8 else 0.0
    decay_slow = decay_ratio > 0.5
    details["decay_ratio"] = round(decay_ratio, 4)
    details["decay_ic_day1"] = round(base_ic_mean, 6)
    details["decay_ic_day5"] = round(ic_day5, 6)

    gates = {
        "param_robust": param_robust,
        "time_stable": time_stable,
        "cap_neutral": cap_neutral,
        "beat_random": beat_random,
        "decay_slow": decay_slow,
    }
    passed = sum(gates.values())

    result = {
        **gates,
        "passed": passed,
        "total": 5,
        "details": details,
    }

    _logger.log("validate", passed=passed, total=5, **gates)

    return result
