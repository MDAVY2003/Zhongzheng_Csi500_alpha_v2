"""WorldQuant 101 单因子横截面分析工具。

输入统一使用 date x code 宽表。日期 t 的因子值与同一行的未来收益配对，
所有 IC、RankIC 和分组收益都在单个交易日的股票横截面上计算。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


__all__ = [
    "calculate_forward_returns",
    "calculate_signal_aligned_open_returns",
    "calculate_daily_ic",
    "calculate_daily_rank_ic",
    "calculate_factor_coverage",
    "summarize_ic",
    "calculate_group_returns",
    "calculate_cumulative_net_value",
    "summarize_long_short_returns",
    "apply_fixed_selection_rules",
    "analyze_single_factor",
    "analyze_factors",
]


DEFAULT_ANNUALIZATION_FACTOR = 252
MIN_VALID_RANK_IC_DAY_RATIO = 0.60
MIN_ADJUSTED_RANK_IC = 0.015
MIN_ADJUSTED_RANK_ICIR = 0.20
MIN_ADJUSTED_RANK_IC_WIN_RATE = 0.52
MIN_POSITIVE_RANK_IC_YEARS = 3
MAX_RANK_IC_TOP_5PCT_ABS_SHARE = 0.30


def _validate_integer_at_least(value: int, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name}必须是大于等于{minimum}的整数")


def _prepare_panel(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """校验宽表，并在不修改原对象的前提下统一日期索引类型。"""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name}必须是pandas DataFrame")
    if frame.shape[1] == 0:
        raise ValueError(f"{name}不能没有股票列")
    if not frame.columns.is_unique:
        raise ValueError(f"{name}的columns存在重复股票代码")

    if isinstance(frame.index, pd.DatetimeIndex):
        date_index = frame.index
    else:
        if is_numeric_dtype(frame.index.dtype):
            raise TypeError(f"{name}的index必须是日期索引或可可靠转换为日期")
        try:
            date_index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"))
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"{name}的index必须是日期索引或可可靠转换为日期"
            ) from error

    if date_index.hasnans:
        raise ValueError(f"{name}的index包含无效日期")
    if not date_index.is_unique:
        raise ValueError(f"{name}的index存在重复日期")
    if not date_index.is_monotonic_increasing:
        raise ValueError(f"{name}的index必须按日期升序排列")

    non_numeric = [
        str(column)
        for column, dtype in frame.dtypes.items()
        if not is_numeric_dtype(dtype)
    ]
    if non_numeric:
        raise TypeError(f"{name}包含非数值列：{non_numeric}")

    values = frame.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError(f"{name}包含正负无穷")

    prepared = frame.copy(deep=False)
    prepared.index = date_index
    return prepared


def _prepare_aligned_pair(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared_factor = _prepare_panel(factor, "factor")
    prepared_returns = _prepare_panel(forward_returns, "forward_returns")

    if not prepared_factor.index.equals(prepared_returns.index):
        raise ValueError("factor与forward_returns的index不一致")
    if not prepared_factor.columns.equals(prepared_returns.columns):
        raise ValueError("factor与forward_returns的columns不一致")
    return prepared_factor, prepared_returns


def _prepare_universe_mask(
    universe_mask: pd.DataFrame | None,
    *,
    index: pd.DatetimeIndex,
    columns: pd.Index,
) -> pd.DataFrame:
    """校验逐日股票池；未提供时默认所有列均属于当日股票池。"""
    if universe_mask is None:
        return pd.DataFrame(True, index=index, columns=columns)
    if not isinstance(universe_mask, pd.DataFrame):
        raise TypeError("universe_mask必须是pandas DataFrame")
    if universe_mask.isna().to_numpy().any():
        raise ValueError("universe_mask不能包含缺失值")
    if not all(is_bool_dtype(dtype) for dtype in universe_mask.dtypes):
        raise TypeError("universe_mask必须全部为布尔列")

    prepared = universe_mask.copy(deep=False)
    if not isinstance(prepared.index, pd.DatetimeIndex):
        prepared.index = pd.DatetimeIndex(
            pd.to_datetime(prepared.index, errors="raise")
        )
    if not prepared.index.equals(index):
        raise ValueError("universe_mask与factor的index不一致")
    if not prepared.columns.equals(columns):
        raise ValueError("universe_mask与factor的columns不一致")
    return prepared


def _calculate_daily_correlation(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    min_obs: int,
    *,
    use_rank: bool,
) -> pd.Series:
    valid = factor.notna() & forward_returns.notna()
    valid_count = valid.sum(axis=1)
    factor_valid = factor.where(valid)
    returns_valid = forward_returns.where(valid)

    if use_rank:
        factor_valid = factor_valid.rank(
            axis=1,
            method="average",
            na_option="keep",
        )
        returns_valid = returns_valid.rank(
            axis=1,
            method="average",
            na_option="keep",
        )
    else:
        # Pearson相关不受正比例缩放影响；先缩放可避免大数因子在平方时溢出。
        factor_scale = factor_valid.abs().max(axis=1).replace(0.0, np.nan)
        returns_scale = returns_valid.abs().max(axis=1).replace(0.0, np.nan)
        factor_valid = factor_valid.div(factor_scale, axis=0)
        returns_valid = returns_valid.div(returns_scale, axis=0)

    correlation = factor_valid.corrwith(returns_valid, axis=1)
    correlation = correlation.where(valid_count >= min_obs)
    correlation = correlation.replace([np.inf, -np.inf], np.nan).astype(float)
    correlation.name = "rank_ic" if use_rank else "ic"
    return correlation


def calculate_forward_returns(
    close: pd.DataFrame,
    periods: int = 1,
) -> pd.DataFrame:
    """计算收盘到未来第 periods 个交易日收盘的收益，并对齐到因子日。

    当 periods=1 时，日期 t 的结果为 close[t+1] / close[t] - 1。
    最后一行没有未来价格，因此保持 NaN，不向前或向后填充。
    """
    _validate_integer_at_least(periods, "periods", 1)
    prepared_close = _prepare_panel(close, "close")
    result = prepared_close.shift(-periods).div(prepared_close) - 1.0
    return result.replace([np.inf, -np.inf], np.nan)


def calculate_signal_aligned_open_returns(
    open_price: pd.DataFrame,
    horizon: int = 1,
) -> pd.DataFrame:
    """计算收盘信号在未来第 ``horizon`` 个持有日的开盘到开盘收益。

    因子在日期 t 收盘后形成，并在 t+1 开盘执行。horizon=1 时，
    日期 t 的收益为 ``open[t+2] / open[t+1] - 1``；horizon=2 时为
    ``open[t+3] / open[t+2] - 1``。位移严格沿输入的全市场交易日索引
    进行，不会跨过单只股票缺失开盘价的日期。
    """
    _validate_integer_at_least(horizon, "horizon", 1)
    prepared_open = _prepare_panel(open_price, "open_price")
    entry_open = prepared_open.shift(-horizon)
    exit_open = prepared_open.shift(-(horizon + 1))
    result = exit_open.div(entry_open) - 1.0
    return result.replace([np.inf, -np.inf], np.nan)


def calculate_daily_ic(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    min_obs: int = 30,
) -> pd.Series:
    """逐日计算因子值与未来收益之间的横截面 Pearson IC。"""
    _validate_integer_at_least(min_obs, "min_obs", 2)
    prepared_factor, prepared_returns = _prepare_aligned_pair(
        factor,
        forward_returns,
    )
    return _calculate_daily_correlation(
        prepared_factor,
        prepared_returns,
        min_obs,
        use_rank=False,
    )


def calculate_daily_rank_ic(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    min_obs: int = 30,
) -> pd.Series:
    """逐日将两组有效值分别排名，再计算横截面 Pearson 相关。"""
    _validate_integer_at_least(min_obs, "min_obs", 2)
    prepared_factor, prepared_returns = _prepare_aligned_pair(
        factor,
        forward_returns,
    )
    return _calculate_daily_correlation(
        prepared_factor,
        prepared_returns,
        min_obs,
        use_rank=True,
    )


def calculate_factor_coverage(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    universe_mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """计算日度联合覆盖率、因子缺失率和有效横截面股票数量。"""
    prepared_factor, prepared_returns = _prepare_aligned_pair(
        factor,
        forward_returns,
    )
    prepared_universe = _prepare_universe_mask(
        universe_mask,
        index=prepared_factor.index,
        columns=prepared_factor.columns,
    )
    universe_count = prepared_universe.sum(axis=1).astype(int)
    factor_valid_count = (
        prepared_factor.notna() & prepared_universe
    ).sum(axis=1)
    valid_count = (
        prepared_factor.notna()
        & prepared_returns.notna()
        & prepared_universe
    ).sum(axis=1)
    valid_denominator = universe_count.where(universe_count > 0)

    return pd.DataFrame(
        {
            "universe_count": universe_count,
            "factor_valid_count": factor_valid_count.astype(int),
            "valid_count": valid_count.astype(int),
            "coverage": valid_count / valid_denominator,
            "missing_rate": 1.0 - factor_valid_count / valid_denominator,
        },
        index=prepared_factor.index,
    )


def _safe_ratio(mean: float, std: float) -> float:
    if pd.isna(mean) or pd.isna(std) or std == 0:
        return np.nan
    value = mean / std
    return float(value) if np.isfinite(value) else np.nan


def _win_rate(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return float(valid.gt(0).mean())


def calculate_cumulative_net_value(returns: pd.Series) -> pd.Series:
    """按有效日复利计算研究型收益差净值，缺失日期保持为NaN。"""
    if not isinstance(returns, pd.Series):
        raise TypeError("returns必须是pandas Series")
    numeric = pd.to_numeric(returns, errors="coerce")
    if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("returns包含正负无穷")
    result = (1.0 + numeric).cumprod(skipna=True)
    result.name = "cumulative_net_value"
    return result


def summarize_long_short_returns(
    long_short_returns: pd.Series,
    *,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> pd.Series:
    """汇总最高组减最低组的研究型收益差，不代表可直接交易的融券组合。"""
    _validate_integer_at_least(
        annualization_factor,
        "annualization_factor",
        1,
    )
    if not isinstance(long_short_returns, pd.Series):
        raise TypeError("long_short_returns必须是pandas Series")

    valid = pd.to_numeric(long_short_returns, errors="coerce").dropna()
    if np.isinf(valid.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("long_short_returns包含正负无穷")
    if valid.empty:
        return pd.Series(
            {
                "long_short_mean": np.nan,
                "long_short_std": np.nan,
                "long_short_annualized_volatility": np.nan,
                "long_short_annualized_sharpe": np.nan,
                "long_short_win_rate": np.nan,
                "long_short_valid_days": 0,
                "long_short_final_net_value": np.nan,
                "long_short_cumulative_return": np.nan,
                "long_short_max_drawdown": np.nan,
            }
        )

    mean = float(valid.mean())
    std = float(valid.std(ddof=1))
    annualized_volatility = (
        std * np.sqrt(annualization_factor) if np.isfinite(std) else np.nan
    )
    daily_sharpe = _safe_ratio(mean, std)
    annualized_sharpe = (
        daily_sharpe * np.sqrt(annualization_factor)
        if not pd.isna(daily_sharpe)
        else np.nan
    )
    net_value = calculate_cumulative_net_value(valid)
    final_net_value = float(net_value.iloc[-1])
    running_peak = net_value.cummax().clip(lower=1.0)
    drawdown = net_value.div(running_peak) - 1.0

    return pd.Series(
        {
            "long_short_mean": mean,
            "long_short_std": std,
            "long_short_annualized_volatility": annualized_volatility,
            "long_short_annualized_sharpe": annualized_sharpe,
            "long_short_win_rate": _win_rate(valid),
            "long_short_valid_days": int(valid.size),
            "long_short_final_net_value": final_net_value,
            "long_short_cumulative_return": final_net_value - 1.0,
            "long_short_max_drawdown": float(drawdown.min()),
        }
    )


def _prefixed(series: pd.Series, prefix: str) -> pd.Series:
    """给汇总指标统一增加前缀。"""
    result = series.copy()
    result.index = [f"{prefix}{name}" for name in result.index]
    return result


def _format_diagnostic_value(value: object) -> str:
    if pd.isna(value):
        return "NaN"
    return f"{float(value):.6f}"


def apply_fixed_selection_rules(summary: pd.Series) -> pd.Series:
    """按照项目固定门槛返回pass、watch或reject及原因。"""
    required = {
        "analysis_status",
        "selected_direction",
        "valid_rank_ic_day_ratio",
        "adjusted_rank_ic",
        "adjusted_rank_icir",
        "adjusted_rank_ic_win_rate",
        "adjusted_positive_rank_ic_years",
        "rank_ic_top_5pct_abs_share",
        "adjusted_long_short_annualized_sharpe",
        "adjusted_group_monotonicity_ratio",
    }
    missing = required.difference(summary.index)
    if missing:
        raise ValueError(f"固定筛选缺少汇总字段：{sorted(missing)}")

    core_checks = {
        "adjusted_rank_ic": (
            pd.notna(summary["adjusted_rank_ic"])
            and summary["adjusted_rank_ic"] >= MIN_ADJUSTED_RANK_IC
        ),
        "adjusted_rank_icir": (
            pd.notna(summary["adjusted_rank_icir"])
            and summary["adjusted_rank_icir"] >= MIN_ADJUSTED_RANK_ICIR
        ),
        "adjusted_rank_ic_win_rate": (
            pd.notna(summary["adjusted_rank_ic_win_rate"])
            and summary["adjusted_rank_ic_win_rate"]
            >= MIN_ADJUSTED_RANK_IC_WIN_RATE
        ),
        "positive_rank_ic_years": (
            pd.notna(summary["adjusted_positive_rank_ic_years"])
            and summary["adjusted_positive_rank_ic_years"]
            >= MIN_POSITIVE_RANK_IC_YEARS
        ),
    }
    outlier_check = (
        pd.notna(summary["rank_ic_top_5pct_abs_share"])
        and summary["rank_ic_top_5pct_abs_share"]
        <= MAX_RANK_IC_TOP_5PCT_ABS_SHARE
    )
    valid_ratio_check = (
        pd.notna(summary["valid_rank_ic_day_ratio"])
        and summary["valid_rank_ic_day_ratio"]
        >= MIN_VALID_RANK_IC_DAY_RATIO
    )
    diagnostic = (
        "诊断：调整后多空Sharpe="
        f"{_format_diagnostic_value(summary['adjusted_long_short_annualized_sharpe'])}，"
        "五组单调率="
        f"{_format_diagnostic_value(summary['adjusted_group_monotonicity_ratio'])}"
    )

    if summary["analysis_status"] != "success":
        status = "reject"
        reason = f"原分析状态为{summary['analysis_status']}；{diagnostic}"
    elif pd.isna(summary["selected_direction"]):
        status = "reject"
        reason = f"无法确定方向；{diagnostic}"
    elif not valid_ratio_check:
        status = "reject"
        reason = (
            "valid_rank_ic_day_ratio低于0.60；"
            f"实际={_format_diagnostic_value(summary['valid_rank_ic_day_ratio'])}；"
            f"{diagnostic}"
        )
    elif not any(core_checks.values()):
        status = "reject"
        reason = f"调整后RankIC、RankICIR、胜率和年度稳定性均未达标；{diagnostic}"
    elif all(core_checks.values()) and outlier_check:
        status = "pass"
        reason = f"通过全部固定筛选；{diagnostic}"
    else:
        failed: list[str] = []
        if not core_checks["adjusted_rank_ic"]:
            failed.append("adjusted_rank_ic<0.015")
        if not core_checks["adjusted_rank_icir"]:
            failed.append("adjusted_rank_icir<0.20")
        if not core_checks["adjusted_rank_ic_win_rate"]:
            failed.append("adjusted_rank_ic_win_rate<0.52")
        if not core_checks["positive_rank_ic_years"]:
            failed.append("调整后年度RankIC为正的年份少于3年")
        if not outlier_check:
            failed.append("RankIC绝对值前5%日期贡献比例>0.30")
        status = "watch"
        reason = f"未通过项目：{'、'.join(failed)}；{diagnostic}"

    return pd.Series(
        {
            "selection_status": status,
            "selection_reason": reason,
        }
    )


def summarize_ic(
    daily_ic: pd.Series,
    daily_rank_ic: pd.Series,
    daily_coverage: pd.Series,
    daily_valid_count: pd.Series,
    *,
    factor_name: str,
    daily_missing_rate: pd.Series | None = None,
) -> pd.Series:
    """汇总日度 IC 序列；标准差使用 pandas 默认样本标准差。"""
    ic_mean = daily_ic.mean()
    ic_std = daily_ic.std(ddof=1)
    rank_ic_mean = daily_rank_ic.mean()
    rank_ic_std = daily_rank_ic.std(ddof=1)

    if pd.isna(rank_ic_mean) or rank_ic_mean == 0:
        suggested_direction: float = np.nan
    else:
        suggested_direction = 1.0 if rank_ic_mean > 0 else -1.0

    return pd.Series(
        {
            "factor_name": factor_name,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": _safe_ratio(ic_mean, ic_std),
            "ic_win_rate": _win_rate(daily_ic),
            "rank_ic_mean": rank_ic_mean,
            "rank_ic_std": rank_ic_std,
            "rank_icir": _safe_ratio(rank_ic_mean, rank_ic_std),
            "rank_ic_win_rate": _win_rate(daily_rank_ic),
            "mean_coverage": daily_coverage.mean(),
            "min_coverage": daily_coverage.min(),
            "mean_missing_rate": (
                daily_missing_rate.mean()
                if daily_missing_rate is not None
                else np.nan
            ),
            "mean_valid_count": daily_valid_count.mean(),
            "min_valid_count": daily_valid_count.min(),
            "valid_days": int(daily_ic.notna().sum()),
            "rank_ic_valid_days": int(daily_rank_ic.notna().sum()),
            "suggested_direction": suggested_direction,
            "direction_note": (
                "该因子可能是反向有效因子"
                if suggested_direction == -1
                else ""
            ),
        }
    )


def calculate_group_returns(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    min_obs: int = 30,
    n_groups: int = 5,
) -> pd.DataFrame:
    """按日将低因子值到高因子值分组，并计算各组等权未来收益。

    大量相同因子值时，使用稳定的股票列顺序打破并列，再按 qcut 的
    线性分位点边界向量化分组，从而保持固定组数且不因重复边界中断。
    """
    _validate_integer_at_least(min_obs, "min_obs", 2)
    _validate_integer_at_least(n_groups, "n_groups", 2)
    prepared_factor, prepared_returns = _prepare_aligned_pair(
        factor,
        forward_returns,
    )

    group_columns = [f"group_{group}" for group in range(1, n_groups + 1)]
    required_count = max(min_obs, n_groups)
    valid = prepared_factor.notna() & prepared_returns.notna()
    valid_count = valid.sum(axis=1)
    unique_factor_count = prepared_factor.where(valid).nunique(axis=1)
    stable_ranks = prepared_factor.where(valid).rank(
        axis=1,
        method="first",
        na_option="keep",
    )

    group_labels = pd.DataFrame(
        1.0,
        index=prepared_factor.index,
        columns=prepared_factor.columns,
    ).where(valid)
    for split in range(1, n_groups):
        # 唯一排名1..n的qcut边界为1 + (n - 1) * split / n_groups。
        boundary = 1.0 + (valid_count - 1.0) * split / n_groups
        group_labels += stable_ranks.gt(boundary, axis=0).astype(float)

    valid_returns = prepared_returns.where(valid)
    result = pd.DataFrame(
        {
            f"group_{group}": valid_returns.where(
                group_labels.eq(group)
            ).mean(axis=1)
            for group in range(1, n_groups + 1)
        },
        index=prepared_factor.index,
        columns=group_columns,
    )
    usable_cross_section = valid_count.ge(required_count) & unique_factor_count.ge(2)
    return result.where(usable_cross_section, np.nan)


def _adjust_group_returns(
    group_returns: pd.DataFrame,
    selected_direction: float,
    n_groups: int,
) -> pd.DataFrame:
    """按选定方向重排组号，调整后第n组始终代表预期高收益端。"""
    adjusted = pd.DataFrame(index=group_returns.index)
    for group in range(1, n_groups + 1):
        if pd.isna(selected_direction):
            adjusted[f"adjusted_group_{group}"] = np.nan
        else:
            source_group = group if selected_direction == 1 else n_groups + 1 - group
            adjusted[f"adjusted_group_{group}"] = group_returns[
                f"group_{source_group}"
            ]
    return adjusted


def _absolute_tail_share(series: pd.Series, fraction: float = 0.05) -> float:
    """返回绝对值最大的若干观测占全部绝对值之和的比例。"""
    valid = series.dropna().abs()
    total = valid.sum()
    if valid.empty or total == 0:
        return np.nan
    tail_count = max(1, int(np.ceil(len(valid) * fraction)))
    return float(valid.nlargest(tail_count).sum() / total)


def _group_diagnostics(
    group_returns: pd.DataFrame,
    suggested_direction: float,
) -> pd.Series:
    """检查各组平均收益是否沿样本内推断方向大致单调。"""
    group_means = group_returns.mean().dropna()
    if len(group_means) < 2:
        return pd.Series(
            {
                "group_return_rank_correlation": np.nan,
                "group_monotonicity_ratio": np.nan,
                "adjusted_group_return_rank_correlation": np.nan,
                "adjusted_group_monotonicity_ratio": np.nan,
                "approximately_monotonic": False,
            }
        )

    group_numbers = pd.Series(
        np.arange(1, len(group_means) + 1, dtype=float),
        index=group_means.index,
    )
    rank_correlation = group_numbers.rank().corr(group_means.rank())
    if pd.isna(suggested_direction):
        monotonicity_ratio = np.nan
    else:
        differences = group_means.diff().dropna()
        monotonicity_ratio = (
            float((differences * suggested_direction > 0).mean())
            if not differences.empty
            else np.nan
        )

    return pd.Series(
        {
            "group_return_rank_correlation": rank_correlation,
            "group_monotonicity_ratio": monotonicity_ratio,
            "adjusted_group_return_rank_correlation": (
                rank_correlation * suggested_direction
                if not pd.isna(suggested_direction)
                else np.nan
            ),
            "adjusted_group_monotonicity_ratio": monotonicity_ratio,
            "approximately_monotonic": bool(
                not pd.isna(monotonicity_ratio)
                and monotonicity_ratio >= 0.75
            ),
        }
    )


def _factor_diagnostics(
    daily_ic: pd.Series,
    daily_rank_ic: pd.Series,
    coverage: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    group_returns: pd.DataFrame,
    *,
    min_obs: int,
    suggested_direction: float,
) -> pd.Series:
    """生成覆盖率、跨年方向、分组单调性和异常日集中度诊断。"""
    max_valid_count = int(coverage["valid_count"].max())
    days_meeting_min_obs = int(coverage["valid_count"].ge(min_obs).sum())
    total_days = int(len(coverage))
    valid_year_means = yearly_summary["rank_ic_mean"].dropna()

    if max_valid_count < min_obs:
        analysis_status = "insufficient_coverage"
    elif daily_ic.notna().sum() == 0 or daily_rank_ic.notna().sum() == 0:
        analysis_status = "no_valid_ic"
    else:
        analysis_status = "success"

    if pd.isna(suggested_direction) or valid_year_means.empty:
        direction_aligned_years = 0
        direction_consistency_rate = np.nan
        direction_stable = False
    else:
        aligned = valid_year_means.mul(suggested_direction).gt(0)
        direction_aligned_years = int(aligned.sum())
        direction_consistency_rate = float(aligned.mean())
        direction_stable = bool(
            len(valid_year_means) >= 2 and direction_consistency_rate >= 0.80
        )

    absolute_year_means = valid_year_means.abs()
    if absolute_year_means.empty or absolute_year_means.sum() == 0:
        dominant_year: float = np.nan
        dominant_year_share = np.nan
    else:
        dominant_year = float(absolute_year_means.idxmax())
        dominant_year_share = float(
            absolute_year_means.max() / absolute_year_means.sum()
        )

    group_diagnostics = _group_diagnostics(
        group_returns,
        suggested_direction,
    )
    return pd.concat(
        [
            pd.Series(
                {
                    "analysis_status": analysis_status,
                    "max_valid_count": max_valid_count,
                    "days_meeting_min_obs": days_meeting_min_obs,
                    "days_meeting_min_obs_rate": (
                        days_meeting_min_obs / total_days if total_days else np.nan
                    ),
                    "valid_years": int(len(valid_year_means)),
                    "direction_aligned_years": direction_aligned_years,
                    "direction_consistency_rate": direction_consistency_rate,
                    "direction_stable": direction_stable,
                    "only_one_direction_aligned_year": bool(
                        len(valid_year_means) > 1
                        and direction_aligned_years == 1
                    ),
                    "dominant_rank_ic_year": dominant_year,
                    "dominant_rank_ic_year_share": dominant_year_share,
                    "ic_top_5pct_abs_share": _absolute_tail_share(daily_ic),
                    "rank_ic_top_5pct_abs_share": _absolute_tail_share(
                        daily_rank_ic
                    ),
                }
            ),
            group_diagnostics,
        ]
    )


def _calculate_yearly_summary(
    daily_ic: pd.Series,
    daily_rank_ic: pd.Series,
    adjusted_daily_ic: pd.Series,
    adjusted_daily_rank_ic: pd.Series,
    coverage: pd.DataFrame,
    long_short_returns: pd.Series,
    adjusted_long_short_returns: pd.Series,
    *,
    selected_direction: float,
    annualization_factor: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for year in sorted(daily_ic.index.year.unique()):
        mask = daily_ic.index.year == year
        year_ic = daily_ic.loc[mask]
        year_rank_ic = daily_rank_ic.loc[mask]
        year_adjusted_ic = adjusted_daily_ic.loc[mask]
        year_adjusted_rank_ic = adjusted_daily_rank_ic.loc[mask]
        year_coverage = coverage.loc[mask, "coverage"]
        year_valid_count = coverage.loc[mask, "valid_count"]
        year_long_short = long_short_returns.loc[mask]
        year_adjusted_long_short = adjusted_long_short_returns.loc[mask]
        ic_mean = year_ic.mean()
        ic_std = year_ic.std(ddof=1)
        rank_ic_mean = year_rank_ic.mean()
        rank_ic_std = year_rank_ic.std(ddof=1)

        row: dict[str, object] = {
            "year": int(year),
            "selected_direction": selected_direction,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": _safe_ratio(ic_mean, ic_std),
            "ic_win_rate": _win_rate(year_ic),
            "rank_ic_mean": rank_ic_mean,
            "rank_ic_std": rank_ic_std,
            "rank_icir": _safe_ratio(rank_ic_mean, rank_ic_std),
            "rank_ic_win_rate": _win_rate(year_rank_ic),
            "adjusted_ic": year_adjusted_ic.mean(),
            "adjusted_ic_std": year_adjusted_ic.std(ddof=1),
            "adjusted_icir": (
                _safe_ratio(ic_mean, ic_std) * selected_direction
                if not pd.isna(selected_direction)
                else np.nan
            ),
            "adjusted_ic_win_rate": _win_rate(year_adjusted_ic),
            "adjusted_rank_ic": year_adjusted_rank_ic.mean(),
            "adjusted_rank_ic_std": year_adjusted_rank_ic.std(ddof=1),
            "adjusted_rank_icir": (
                _safe_ratio(rank_ic_mean, rank_ic_std) * selected_direction
                if not pd.isna(selected_direction)
                else np.nan
            ),
            "adjusted_rank_ic_win_rate": _win_rate(year_adjusted_rank_ic),
            "mean_coverage": year_coverage.mean(),
            "min_coverage": year_coverage.min(),
            "mean_valid_count": year_valid_count.mean(),
            "min_valid_count": int(year_valid_count.min()),
            "max_valid_count": int(year_valid_count.max()),
            "valid_days": int(year_ic.notna().sum()),
            "rank_ic_valid_days": int(year_rank_ic.notna().sum()),
        }
        row.update(
            summarize_long_short_returns(
                year_long_short,
                annualization_factor=annualization_factor,
            ).to_dict()
        )
        row.update(
            _prefixed(
                summarize_long_short_returns(
                    year_adjusted_long_short,
                    annualization_factor=annualization_factor,
                ),
                "adjusted_",
            ).to_dict()
        )
        rows.append(row)

    return pd.DataFrame(rows).set_index("year")


def analyze_single_factor(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    factor_name: str,
    min_obs: int = 30,
    n_groups: int = 5,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
    universe_mask: pd.DataFrame | None = None,
) -> dict[str, object]:
    """完成一个原始方向因子的 IC、覆盖率和分组收益分析。"""
    if not isinstance(factor_name, str) or not factor_name.strip():
        raise ValueError("factor_name不能为空")
    _validate_integer_at_least(min_obs, "min_obs", 2)
    _validate_integer_at_least(n_groups, "n_groups", 2)
    _validate_integer_at_least(
        annualization_factor,
        "annualization_factor",
        1,
    )
    prepared_factor, prepared_returns = _prepare_aligned_pair(
        factor,
        forward_returns,
    )
    prepared_universe = _prepare_universe_mask(
        universe_mask,
        index=prepared_factor.index,
        columns=prepared_factor.columns,
    )
    prepared_factor = prepared_factor.where(prepared_universe)
    prepared_returns = prepared_returns.where(prepared_universe)

    coverage = calculate_factor_coverage(
        prepared_factor,
        prepared_returns,
        universe_mask=prepared_universe,
    )
    daily_ic = _calculate_daily_correlation(
        prepared_factor,
        prepared_returns,
        min_obs,
        use_rank=False,
    )
    daily_rank_ic = _calculate_daily_correlation(
        prepared_factor,
        prepared_returns,
        min_obs,
        use_rank=True,
    )
    ic_summary = summarize_ic(
        daily_ic,
        daily_rank_ic,
        coverage["coverage"],
        coverage["valid_count"],
        factor_name=factor_name,
        daily_missing_rate=coverage["missing_rate"],
    )
    selected_direction = ic_summary["suggested_direction"]
    adjusted_daily_ic = daily_ic * selected_direction
    adjusted_daily_ic.name = "adjusted_ic"
    adjusted_daily_rank_ic = daily_rank_ic * selected_direction
    adjusted_daily_rank_ic.name = "adjusted_rank_ic"

    group_returns = calculate_group_returns(
        prepared_factor,
        prepared_returns,
        min_obs=min_obs,
        n_groups=n_groups,
    )
    adjusted_group_returns = _adjust_group_returns(
        group_returns,
        selected_direction,
        n_groups,
    )
    long_short_returns = (
        group_returns[f"group_{n_groups}"] - group_returns["group_1"]
    )
    long_short_returns.name = "long_short_return"
    adjusted_long_short_returns = long_short_returns * selected_direction
    adjusted_long_short_returns.name = "adjusted_long_short_return"
    long_short_cumulative_net_value = calculate_cumulative_net_value(
        long_short_returns
    )
    adjusted_long_short_cumulative_net_value = calculate_cumulative_net_value(
        adjusted_long_short_returns
    )
    adjusted_long_short_cumulative_net_value.name = (
        "adjusted_cumulative_net_value"
    )

    long_short_summary = summarize_long_short_returns(
        long_short_returns,
        annualization_factor=annualization_factor,
    )
    adjusted_long_short_summary = _prefixed(
        summarize_long_short_returns(
            adjusted_long_short_returns,
            annualization_factor=annualization_factor,
        ),
        "adjusted_",
    )
    yearly_summary = _calculate_yearly_summary(
        daily_ic,
        daily_rank_ic,
        adjusted_daily_ic,
        adjusted_daily_rank_ic,
        coverage,
        long_short_returns,
        adjusted_long_short_returns,
        selected_direction=selected_direction,
        annualization_factor=annualization_factor,
    )
    diagnostics = _factor_diagnostics(
        daily_ic,
        daily_rank_ic,
        coverage,
        yearly_summary,
        group_returns,
        min_obs=min_obs,
        suggested_direction=selected_direction,
    )
    eligible_rank_ic_days = int(
        (prepared_returns.notna() & prepared_universe)
        .sum(axis=1)
        .ge(min_obs)
        .sum()
    )
    valid_rank_ic_day_ratio = (
        int(daily_rank_ic.notna().sum()) / eligible_rank_ic_days
        if eligible_rank_ic_days
        else np.nan
    )
    direction_summary = pd.Series(
        {
            "selected_direction": selected_direction,
            "adjusted_ic": ic_summary["ic_mean"] * selected_direction,
            "adjusted_ic_std": ic_summary["ic_std"],
            "adjusted_icir": ic_summary["icir"] * selected_direction,
            "adjusted_ic_win_rate": _win_rate(adjusted_daily_ic),
            "adjusted_rank_ic": (
                ic_summary["rank_ic_mean"] * selected_direction
            ),
            "adjusted_rank_ic_std": ic_summary["rank_ic_std"],
            "adjusted_rank_icir": (
                ic_summary["rank_icir"] * selected_direction
            ),
            "adjusted_rank_ic_win_rate": _win_rate(
                adjusted_daily_rank_ic
            ),
            "eligible_rank_ic_days": eligible_rank_ic_days,
            "valid_rank_ic_day_ratio": valid_rank_ic_day_ratio,
            "adjusted_positive_rank_ic_years": (
                int(yearly_summary["adjusted_rank_ic"].gt(0).sum())
                if not pd.isna(selected_direction)
                else np.nan
            ),
        }
    )
    summary = pd.concat(
        [
            ic_summary,
            direction_summary,
            long_short_summary,
            adjusted_long_short_summary,
            diagnostics,
        ]
    )
    selection = apply_fixed_selection_rules(summary)
    summary = pd.concat([summary, selection])

    return {
        "summary": summary,
        "daily_ic": daily_ic,
        "daily_rank_ic": daily_rank_ic,
        "adjusted_daily_ic": adjusted_daily_ic,
        "adjusted_daily_rank_ic": adjusted_daily_rank_ic,
        "daily_coverage": coverage["coverage"],
        "daily_missing_rate": coverage["missing_rate"],
        "daily_valid_count": coverage["valid_count"],
        "coverage_details": coverage,
        "yearly_summary": yearly_summary,
        "group_returns": group_returns,
        "adjusted_group_returns": adjusted_group_returns,
        "long_short_returns": long_short_returns,
        "adjusted_long_short_returns": adjusted_long_short_returns,
        "long_short_cumulative_net_value": long_short_cumulative_net_value,
        "adjusted_long_short_cumulative_net_value": (
            adjusted_long_short_cumulative_net_value
        ),
    }


def analyze_factors(
    factors: Mapping[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    *,
    min_obs: int = 30,
    n_groups: int = 5,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
    universe_mask: pd.DataFrame | None = None,
) -> dict[str, object]:
    """批量分析多个因子，返回总汇总表和逐因子详细结果。"""
    if not isinstance(factors, Mapping) or not factors:
        raise ValueError("factors必须是非空的因子名称到DataFrame映射")

    details: dict[str, dict[str, object]] = {}
    summaries: list[pd.Series] = []
    for factor_name, factor in factors.items():
        analysis = analyze_single_factor(
            factor,
            forward_returns,
            factor_name=factor_name,
            min_obs=min_obs,
            n_groups=n_groups,
            annualization_factor=annualization_factor,
            universe_mask=universe_mask,
        )
        details[factor_name] = analysis
        summaries.append(analysis["summary"])

    summary = pd.DataFrame(summaries).reset_index(drop=True)
    return {"summary": summary, "details": details}
