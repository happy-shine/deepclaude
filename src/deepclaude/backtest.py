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
