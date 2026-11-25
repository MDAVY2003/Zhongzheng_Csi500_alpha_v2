"""单因子信号的 RankIC 衰减与连续半衰寿命计算。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import factor_analysis as fa


def calculate_delayed_peak_effective_window(
    decay: pd.DataFrame,
    *,
    factor_name: str,
    peak_horizon: int,
    peak_rankic: float,
    max_horizon: int = 10,
    comparison_tolerance: float = 1e-12,
) -> dict[str, object]:
    """计算包含延迟峰值的最大连续半高宽区间。

    从峰值向左右逐点扩展，第一次低于峰值一半时立即停止。即使更远的
    horizon 再次反弹，也不会跨过已跌破的点重新纳入有效区间。
    """
    required = {"horizon", "mean_rank_ic"}
    missing = required.difference(decay.columns)
    if missing:
        raise ValueError(f"decay缺少字段：{sorted(missing)}")
    if isinstance(peak_horizon, bool) or not isinstance(peak_horizon, int):
        raise TypeError("peak_horizon必须是整数")
    if not 1 <= peak_horizon <= max_horizon:
        raise ValueError("peak_horizon必须位于D1至最大horizon之间")
    if not np.isfinite(peak_rankic) or peak_rankic <= 0:
        raise ValueError("peak_rankic必须是有限正数")
    if comparison_tolerance < 0:
        raise ValueError("comparison_tolerance不能为负数")

    ordered = decay.loc[:, ["horizon", "mean_rank_ic"]].copy()
    ordered["horizon"] = pd.to_numeric(ordered["horizon"], errors="raise")
    if not np.equal(ordered["horizon"], ordered["horizon"].astype(int)).all():
        raise ValueError("horizon必须是整数")
    ordered["horizon"] = ordered["horizon"].astype(int)
    if ordered["horizon"].duplicated().any():
        raise ValueError("decay存在重复horizon")
    ordered = ordered.sort_values("horizon")
    expected = list(range(1, max_horizon + 1))
    if ordered["horizon"].tolist() != expected:
        raise ValueError(f"decay必须完整包含D1-D{max_horizon}")
    values = pd.to_numeric(
        ordered.set_index("horizon")["mean_rank_ic"], errors="coerce"
    )
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("D1-D10 RankIC必须全部是有限值")

    observed_peak = float(values.loc[peak_horizon])
    if not np.isclose(
        observed_peak,
        float(peak_rankic),
        rtol=0.0,
        atol=comparison_tolerance,
    ):
        raise ValueError(
            f"{factor_name}的peak_rankic与decay汇总不一致："
            f"selected={peak_rankic}, summary={observed_peak}"
        )
    threshold = 0.5 * observed_peak

    start = peak_horizon
    for horizon in range(peak_horizon - 1, 0, -1):
        if float(values.loc[horizon]) + comparison_tolerance < threshold:
            break
        start = horizon

    end = peak_horizon
    for horizon in range(peak_horizon + 1, max_horizon + 1):
        if float(values.loc[horizon]) + comparison_tolerance < threshold:
            break
        end = horizon

    horizons = list(range(start, end + 1))
    result = {
        "factor": factor_name,
        "peak_horizon": peak_horizon,
        "peak_rankic": observed_peak,
        "half_peak_threshold": threshold,
        "effective_start_horizon": start,
        "effective_end_horizon": end,
        "effective_window_length": end - start + 1,
        "left_censored": start == 1,
        "right_censored": end == max_horizon,
        "effective_horizons": ",".join(f"D{horizon}" for horizon in horizons),
    }
    audit_delayed_peak_effective_window(
        result,
        values,
        max_horizon=max_horizon,
        comparison_tolerance=comparison_tolerance,
    )
    return result


def audit_delayed_peak_effective_window(
    result: dict[str, object],
    values: pd.Series,
    *,
    max_horizon: int = 10,
    comparison_tolerance: float = 1e-12,
) -> None:
    """审计峰值中心有效区间的连续性、阈值和相邻边界。"""
    peak = int(result["peak_horizon"])
    start = int(result["effective_start_horizon"])
    end = int(result["effective_end_horizon"])
    threshold = float(result["half_peak_threshold"])
    if not start <= peak <= end:
        raise AssertionError("peak_horizon不在有效区间内")
    if int(result["effective_window_length"]) != end - start + 1:
        raise AssertionError("effective_window_length计算错误")

    expected_horizons = ",".join(f"D{horizon}" for horizon in range(start, end + 1))
    if result["effective_horizons"] != expected_horizons:
        raise AssertionError("effective_horizons不是连续区间")
    inside = values.loc[start:end].astype(float)
    if (inside + comparison_tolerance < threshold).any():
        raise AssertionError("有效区间内存在低于半峰阈值的horizon")
    if start > 1 and not float(values.loc[start - 1]) + comparison_tolerance < threshold:
        raise AssertionError("左侧区间外相邻horizon未低于半峰阈值")
    if end < max_horizon and not float(values.loc[end + 1]) + comparison_tolerance < threshold:
        raise AssertionError("右侧区间外相邻horizon未低于半峰阈值")
    if bool(result["left_censored"]) != (start == 1):
        raise AssertionError("left_censored与左边界不一致")
    if bool(result["right_censored"]) != (end == max_horizon):
        raise AssertionError("right_censored与右边界不一致")


def _validate_aligned_panels(
    factor: pd.DataFrame,
    open_price: pd.DataFrame,
    universe_mask: pd.DataFrame,
) -> None:
    for frame, name in (
        (factor, "factor"),
        (open_price, "open_price"),
        (universe_mask, "universe_mask"),
    ):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name}必须是pd.DataFrame")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(f"{name}必须使用DatetimeIndex")
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError(f"{name}的日期必须唯一并升序")
        if not frame.columns.is_unique:
            raise ValueError(f"{name}的股票代码列不能重复")
    if not factor.columns.equals(open_price.columns):
        raise ValueError("factor与open_price的columns不一致")
    if not universe_mask.columns.equals(open_price.columns):
        raise ValueError("universe_mask与open_price的columns不一致")
    if not factor.index.isin(open_price.index).all():
        raise ValueError("factor包含open_price交易日历之外的日期")
    if not universe_mask.index.equals(factor.index):
        raise ValueError("factor与universe_mask的index不一致")


def calculate_rankic_decay_horizon(
    factor: pd.DataFrame,
    open_price: pd.DataFrame,
    universe_mask: pd.DataFrame,
    *,
    factor_name: str,
    direction: float,
    horizon: int,
    min_obs: int = 30,
) -> tuple[pd.Series, pd.Series, dict[str, object]]:
    """计算固定收盘信号在一个未来持有日上的日度 RankIC 与汇总。"""
    _validate_aligned_panels(factor, open_price, universe_mask)
    if direction not in {-1, 1, -1.0, 1.0}:
        raise ValueError("direction必须是1或-1")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon必须是正整数")
    if isinstance(min_obs, bool) or not isinstance(min_obs, int) or min_obs < 2:
        raise ValueError("min_obs必须大于等于2")

    returns = fa.calculate_signal_aligned_open_returns(
        open_price,
        horizon=horizon,
    ).loc[factor.index]
    return calculate_rankic_decay_from_returns(
        factor,
        returns,
        universe_mask,
        factor_name=factor_name,
        direction=direction,
        horizon=horizon,
        min_obs=min_obs,
    )


def calculate_rankic_decay_from_returns(
    factor: pd.DataFrame,
    future_returns: pd.DataFrame,
    universe_mask: pd.DataFrame,
    *,
    factor_name: str,
    direction: float,
    horizon: int,
    min_obs: int = 30,
) -> tuple[pd.Series, pd.Series, dict[str, object]]:
    """使用预先构造的未来收益计算一个horizon的RankIC。"""
    if not factor.index.equals(future_returns.index):
        raise ValueError("factor与future_returns的index不一致")
    if not factor.columns.equals(future_returns.columns):
        raise ValueError("factor与future_returns的columns不一致")
    if not universe_mask.index.equals(factor.index):
        raise ValueError("factor与universe_mask的index不一致")
    if not universe_mask.columns.equals(factor.columns):
        raise ValueError("factor与universe_mask的columns不一致")
    if direction not in {-1, 1, -1.0, 1.0}:
        raise ValueError("direction必须是1或-1")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon必须是正整数")
    if isinstance(min_obs, bool) or not isinstance(min_obs, int) or min_obs < 2:
        raise ValueError("min_obs必须大于等于2")

    adjusted_factor = factor * float(direction)
    masked_factor = adjusted_factor.where(universe_mask)
    masked_returns = future_returns.where(universe_mask)
    jointly_valid = masked_factor.notna() & masked_returns.notna()
    valid_count = jointly_valid.sum(axis=1).astype("int64")
    eligible_days = int(
        (masked_returns.notna() & universe_mask).sum(axis=1).ge(min_obs).sum()
    )
    daily_rank_ic = fa.calculate_daily_rank_ic(
        masked_factor,
        masked_returns,
        min_obs=min_obs,
    )
    valid_days = int(daily_rank_ic.notna().sum())
    valid_rank_ic = daily_rank_ic.dropna()
    mean_rank_ic = daily_rank_ic.mean()
    rank_ic_std = daily_rank_ic.std(ddof=1)
    rank_icir = (
        mean_rank_ic / rank_ic_std
        if pd.notna(rank_ic_std) and rank_ic_std != 0
        else np.nan
    )
    summary = {
        "factor": factor_name,
        "horizon": horizon,
        "mean_rank_ic": mean_rank_ic,
        "rank_ic_std": rank_ic_std,
        "rank_icir": rank_icir,
        "win_rate": (
            float(valid_rank_ic.gt(0).mean())
            if not valid_rank_ic.empty
            else np.nan
        ),
        "valid_days": valid_days,
        "eligible_days": eligible_days,
        "valid_day_ratio": valid_days / eligible_days if eligible_days else np.nan,
        "mean_n_stocks": valid_count.mean(),
        "min_n_stocks": int(valid_count.min()),
    }
    daily_rank_ic.name = factor_name
    valid_count.name = factor_name
    return daily_rank_ic, valid_count, summary


def classify_decay_curve(
    decay: pd.DataFrame,
    *,
    factor_name: str,
    direction: float,
    comparison_tolerance: float = 1e-12,
    near_zero: float = 1e-12,
) -> dict[str, object]:
    """将D1-D10曲线分为普通衰减、延迟峰值或D1异常。"""
    required = {"horizon", "mean_rank_ic"}
    missing = required.difference(decay.columns)
    if missing:
        raise ValueError(f"decay缺少字段：{sorted(missing)}")
    ordered = decay.sort_values("horizon").reset_index(drop=True)
    horizons = ordered["horizon"].astype(int).tolist()
    if horizons != list(range(1, 11)):
        raise ValueError("decay必须完整包含D1-D10")

    values = ordered.set_index("horizon")["mean_rank_ic"].astype(float)
    d1 = values.loc[1]
    base: dict[str, object] = {
        "factor": factor_name,
        "direction": direction,
        **{f"D{horizon}": values.loc[horizon] for horizon in range(1, 11)},
    }
    if pd.isna(direction):
        return {
            **base,
            "decay_type": "d1_invalid",
            "anomaly_reason": "direction_missing",
        }
    if pd.isna(d1):
        return {
            **base,
            "decay_type": "d1_invalid",
            "anomaly_reason": "d1_nan_or_no_valid_rankic",
        }
    if abs(float(d1)) <= near_zero:
        return {
            **base,
            "decay_type": "d1_invalid",
            "anomaly_reason": "d1_near_zero",
        }
    if float(d1) < 0:
        return {
            **base,
            "decay_type": "d1_invalid",
            "anomaly_reason": "adjusted_d1_negative",
        }
    later = values.loc[2:10]
    delayed_peak = bool((later > float(d1) + comparison_tolerance).any())
    return {
        **base,
        "decay_type": "delayed_peak" if delayed_peak else "normal",
        "anomaly_reason": "",
    }


def summarize_normal_decay(
    classified: dict[str, object],
) -> dict[str, object]:
    """按D1半阈值计算普通衰减因子的连续寿命。"""
    if classified.get("decay_type") != "normal":
        raise ValueError("仅normal曲线可以计算普通寿命")
    d1 = float(classified["D1"])
    threshold = 0.5 * d1
    result = dict(classified)
    result["half_threshold"] = threshold
    for horizon in range(2, 11):
        value = classified[f"D{horizon}"]
        if pd.isna(value):
            result.update(
                {
                    "lifetime": np.nan,
                    "lifetime_lower_bound": horizon - 1,
                    "censored": True,
                    "lifetime_status": "boundary_censored",
                }
            )
            return result
        if float(value) < threshold:
            result.update(
                {
                    "lifetime": horizon - 1,
                    "lifetime_lower_bound": horizon - 1,
                    "censored": False,
                    "lifetime_status": "determined",
                }
            )
            return result
    result.update(
        {
            "lifetime": np.nan,
            "lifetime_lower_bound": 10,
            "censored": True,
            "lifetime_status": "right_censored",
        }
    )
    return result


def summarize_delayed_peak(
    classified: dict[str, object],
) -> dict[str, object]:
    """计算延迟峰值位置及峰值之后的半衰寿命。"""
    if classified.get("decay_type") != "delayed_peak":
        raise ValueError("仅delayed_peak曲线可以计算峰后寿命")
    values = pd.Series(
        {horizon: classified[f"D{horizon}"] for horizon in range(1, 11)},
        dtype=float,
    )
    peak_horizon = int(values.idxmax())
    peak_rankic = float(values.loc[peak_horizon])
    d1 = float(values.loc[1])
    threshold = 0.5 * peak_rankic
    result = dict(classified)
    result.update(
        {
            "peak_horizon": peak_horizon,
            "peak_rankic": peak_rankic,
            "peak_vs_d1_ratio": peak_rankic / d1,
            "peak_half_threshold": threshold,
        }
    )
    for horizon in range(peak_horizon + 1, 11):
        value = values.loc[horizon]
        if pd.isna(value):
            result.update(
                {
                    "post_peak_lifetime": np.nan,
                    "post_peak_lifetime_lower_bound": horizon - peak_horizon - 1,
                    "censored": True,
                    "post_peak_lifetime_status": "boundary_censored",
                }
            )
            return result
        if float(value) < threshold:
            result.update(
                {
                    "post_peak_lifetime": horizon - peak_horizon,
                    "post_peak_lifetime_lower_bound": horizon - peak_horizon,
                    "censored": False,
                    "post_peak_lifetime_status": "determined",
                }
            )
            return result
    result.update(
        {
            "post_peak_lifetime": np.nan,
            "post_peak_lifetime_lower_bound": 10 - peak_horizon,
            "censored": True,
            "post_peak_lifetime_status": "right_censored",
        }
    )
    return result


def summarize_factor_lifetime(
    decay: pd.DataFrame,
    *,
    factor_name: str,
    direction: float,
    near_zero: float = 1e-12,
) -> dict[str, object]:
    """按首次跌破 D1 RankIC 一半的位置确定连续信号寿命。"""
    required = {"horizon", "mean_rank_ic"}
    missing = required.difference(decay.columns)
    if missing:
        raise ValueError(f"decay缺少字段：{sorted(missing)}")
    ordered = decay.sort_values("horizon").reset_index(drop=True)
    if ordered.empty or int(ordered.loc[0, "horizon"]) != 1:
        raise ValueError("decay必须从D1开始")
    horizons = ordered["horizon"].astype(int).tolist()
    if horizons != list(range(1, max(horizons) + 1)):
        raise ValueError("decay的horizon必须从1开始连续")

    d1 = float(ordered.loc[0, "mean_rank_ic"])
    base = {
        "factor": factor_name,
        "direction": float(direction),
        "mean_rank_ic_d1": d1,
        "half_threshold": 0.5 * d1 if math.isfinite(d1) else np.nan,
    }
    if not math.isfinite(d1) or d1 <= near_zero:
        return {
            **base,
            "lifetime": np.nan,
            "lifetime_lower_bound": 0,
            "rank_ic_at_lifetime": np.nan,
            "previous_day_rank_ic": np.nan,
            "first_below_horizon": np.nan,
            "first_below_rank_ic": np.nan,
            "censored": False,
            "lifetime_status": "invalid_d1",
        }

    threshold = 0.5 * d1
    values = ordered.set_index("horizon")["mean_rank_ic"]
    for horizon in horizons[1:]:
        value = values.loc[horizon]
        if pd.isna(value):
            lower_bound = horizon - 1
            return {
                **base,
                "lifetime": np.nan,
                "lifetime_lower_bound": lower_bound,
                "rank_ic_at_lifetime": float(values.loc[lower_bound]),
                "previous_day_rank_ic": (
                    float(values.loc[lower_bound - 1])
                    if lower_bound > 1
                    else np.nan
                ),
                "first_below_horizon": np.nan,
                "first_below_rank_ic": np.nan,
                "censored": True,
                "lifetime_status": "boundary_censored",
            }
        if value < threshold:
            lifetime = horizon - 1
            return {
                **base,
                "lifetime": lifetime,
                "lifetime_lower_bound": lifetime,
                "rank_ic_at_lifetime": float(values.loc[lifetime]),
                "previous_day_rank_ic": (
                    float(values.loc[lifetime - 1])
                    if lifetime > 1
                    else np.nan
                ),
                "first_below_horizon": horizon,
                "first_below_rank_ic": float(value),
                "censored": False,
                "lifetime_status": "determined",
            }

    lower_bound = horizons[-1]
    return {
        **base,
        "lifetime": np.nan,
        "lifetime_lower_bound": lower_bound,
        "rank_ic_at_lifetime": float(values.loc[lower_bound]),
        "previous_day_rank_ic": (
            float(values.loc[lower_bound - 1]) if lower_bound > 1 else np.nan
        ),
        "first_below_horizon": np.nan,
        "first_below_rank_ic": np.nan,
        "censored": True,
        "lifetime_status": "right_censored",
    }
