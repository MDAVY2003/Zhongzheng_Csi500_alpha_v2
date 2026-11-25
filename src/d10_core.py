"""Ex-ante D10 construction and lifetime helpers (v3).

Key rule: the ranking universe is `membership & signal.notna()`.  Forward
returns are never allowed to change D10 membership.  The equal-weight
benchmark uses all members with a valid forward return, independent of the
factor value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def nw_tstat(series: pd.Series, lag: int) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) < max(20, lag + 2):
        return np.nan
    e = (x - x.mean()).to_numpy(dtype=float)
    t = len(e)
    gamma0 = float(np.mean(e * e))
    var = gamma0
    for ell in range(1, lag + 1):
        gamma = float(np.mean(e[ell:] * e[:-ell]))
        var += 2.0 * (1.0 - ell / (lag + 1.0)) * gamma
    if not np.isfinite(var) or var <= 0:
        return np.nan
    return float(x.mean() / math.sqrt(var / t))


@dataclass(slots=True)
class HorizonStats:
    d10: float
    d1: float
    rankic: float
    coverage: float
    n_days: int
    missing_d10: int
    d10_series: pd.Series
    d1_series: pd.Series
    ew_series: pd.Series


def compute_horizon(
    signal: pd.DataFrame,
    fwd: pd.DataFrame,
    mem: pd.DataFrame,
    h: int,
) -> HorizonStats:
    """D10 is fixed ex-ante; missing forward returns do not change membership."""
    signal_valid = mem & signal.notna()
    if not signal_valid.to_numpy().any():
        empty = pd.Series(np.nan, index=signal.index, dtype=float)
        return HorizonStats(np.nan, np.nan, np.nan, 0.0, 0, 0, empty, empty, empty)

    pct = signal.where(signal_valid).rank(axis=1, method="first", pct=True)
    dec = np.ceil(pct * 10).clip(1, 10)
    d10_mask = dec.eq(10)
    d1_mask = dec.eq(1)

    # Benchmark is all members with a valid forward return, not factor-valid.
    return_valid = mem & fwd.notna()
    ew = fwd.where(return_valid).mean(axis=1)
    excess = fwd.where(return_valid).sub(ew, axis=0)

    d10_series = excess.where(d10_mask).mean(axis=1)
    d1_series = excess.where(d1_mask).mean(axis=1)
    missing_d10 = int((d10_mask & ~fwd.notna()).to_numpy().sum())

    both = signal_valid & return_valid
    ic = signal.where(both).rank(axis=1).corrwith(
        fwd.where(both).rank(axis=1), axis=1
    ).dropna()

    n_days = int(d10_series.notna().sum())
    d10 = float(d10_series.mean() * 252 / h) if n_days else np.nan
    d1 = float(d1_series.mean() * 252 / h) if d1_series.notna().any() else np.nan
    return HorizonStats(
        d10=d10,
        d1=d1,
        rankic=float(ic.mean()) if len(ic) else np.nan,
        coverage=float(signal_valid.to_numpy().sum() / mem.to_numpy().sum()),
        n_days=n_days,
        missing_d10=missing_d10,
        d10_series=d10_series,
        d1_series=d1_series,
        ew_series=ew,
    )


def curve_for_direction(
    raw_signal: pd.DataFrame,
    fwd_by_h: dict[int, pd.DataFrame],
    mem: pd.DataFrame,
    horizons: list[int],
    direction: float,
) -> dict[int, HorizonStats]:
    sig = raw_signal * float(direction)
    return {h: compute_horizon(sig, fwd_by_h[h], mem, h) for h in horizons}


def choose_direction(
    raw_signal: pd.DataFrame,
    fwd_by_h: dict[int, pd.DataFrame],
    mem: pd.DataFrame,
    horizons: list[int],
) -> tuple[float, dict[int, HorizonStats], dict[int, HorizonStats]]:
    pos = curve_for_direction(raw_signal, fwd_by_h, mem, horizons, +1.0)
    neg = curve_for_direction(raw_signal, fwd_by_h, mem, horizons, -1.0)
    pos_peak = max((s.d10 for s in pos.values() if np.isfinite(s.d10)), default=np.nan)
    neg_peak = max((s.d10 for s in neg.values() if np.isfinite(s.d10)), default=np.nan)
    if not np.isfinite(pos_peak) and not np.isfinite(neg_peak):
        return np.nan, pos, neg
    if np.isfinite(pos_peak) and (not np.isfinite(neg_peak) or pos_peak >= neg_peak):
        return 1.0, pos, neg
    return -1.0, pos, neg
