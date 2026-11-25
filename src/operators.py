"""
WorldQuant 101 基础算子。

统一数据约定：
- 输入宽表的行是按升序排列的交易日期；
- 输入宽表的列是股票代码；
- rank 和 scale 沿列方向执行，即每天做一次横截面计算；
- 其他 ts/rolling/delay/delta 算子沿行方向执行，即每只股票独立计算。

从长表构造输入示例：

    close = bars.pivot(index="date", columns="code", values="close").sort_index()
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator
from typing import TypeVar

import numpy as np
import pandas as pd


PandasObject = TypeVar("PandasObject", pd.Series, pd.DataFrame)

__all__ = [
    "rank",
    "ts_rank",
    "delay",
    "delta",
    "rolling_mean",
    "rolling_sum",
    "rolling_std",
    "rolling_correlation",
    "rolling_covariance",
    "ts_min",
    "ts_max",
    "ts_argmax",
    "ts_argmin",
    "scale",
    "sign",
    "abs",
    "decay_linear",
    "elementwise_max",
    "elementwise_min",
    "industry_neutralize",
    "log",
    "rolling_product",
    "signed_power",
    "use_universe_mask",
]


_UNIVERSE_MASK: ContextVar[pd.DataFrame | None] = ContextVar(
    "alpha_universe_mask",
    default=None,
)


def _validate_time_input(data: PandasObject, name: str) -> None:
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(f"{name} 只接受 pandas Series 或 DataFrame")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError(f"{name} 要求索引为 pandas DatetimeIndex")
    if not data.index.is_unique:
        raise ValueError(f"{name} 要求日期索引不重复")
    if not data.index.is_monotonic_increasing:
        raise ValueError(f"{name} 要求日期索引按升序排列")
    if isinstance(data, pd.DataFrame) and not data.columns.is_unique:
        raise ValueError(f"{name} 要求股票代码列不重复")


def _validate_panel(frame: pd.DataFrame, name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"{name} 是横截面算子，要求 date×code 形式的 pandas DataFrame"
        )
    _validate_time_input(frame, name)


def _apply_universe_mask(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """横截面算子只保留上下文指定的当日股票池。"""
    mask = _UNIVERSE_MASK.get()
    if mask is None:
        return frame
    if not frame.index.equals(mask.index) or not frame.columns.equals(mask.columns):
        raise ValueError(f"{name}的输入与历史股票池掩码不一致")
    return frame.where(mask)


@contextmanager
def use_universe_mask(mask: pd.DataFrame | None) -> Iterator[None]:
    """临时设置逐日股票池，供rank、scale和行业中性化使用。"""
    if mask is not None:
        _validate_panel(mask, "universe_mask")
        prepared = mask.fillna(False).astype(bool)
    else:
        prepared = None
    token = _UNIVERSE_MASK.set(prepared)
    try:
        yield
    finally:
        _UNIVERSE_MASK.reset(token)


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数")


def _validate_nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} 必须是非负整数")


def _validate_pair(
    left: PandasObject,
    right: PandasObject,
    name: str,
) -> None:
    _validate_time_input(left, name)
    _validate_time_input(right, name)
    if type(left) is not type(right):
        raise TypeError(f"{name} 的两个输入必须同为 Series 或同为 DataFrame")
    if not left.index.equals(right.index):
        raise ValueError(f"{name} 的两个输入日期索引必须完全一致")
    if isinstance(left, pd.DataFrame) and not left.columns.equals(right.columns):
        raise ValueError(f"{name} 的两个输入股票代码列必须完全一致")


def rank(frame: pd.DataFrame) -> pd.DataFrame:
    """
    每个交易日对全部股票进行横截面百分位排名。

    结果范围为 (0, 1]；并列值取平均排名，缺失值保持缺失。
    """
    _validate_panel(frame, "rank")
    prepared = _apply_universe_mask(frame, "rank")
    return prepared.rank(axis=1, method="average", pct=True, na_option="keep")


def ts_rank(data: PandasObject, window: int) -> PandasObject:
    """
    对每只股票计算当前值在过去 window 个观测中的时间序列排名。

    排名范围为 [1/window, 1]，与横截面 rank 使用同一百分位尺度。
    并列值取平均排名。窗口未满或包含缺失值时结果为 NaN。
    """
    _validate_time_input(data, "ts_rank")
    _validate_positive_integer(window, "window")

    def rank_last(values: np.ndarray) -> float:
        last_value = values[-1]
        less_count = np.sum(values < last_value)
        equal_count = np.sum(values == last_value)
        return float(less_count + (equal_count + 1) / 2) / len(values)

    return data.rolling(window=window, min_periods=window).apply(
        rank_last,
        raw=True,
    )


def delay(data: PandasObject, period: int = 1) -> PandasObject:
    """每只股票向后滞后 period 个观测。"""
    _validate_time_input(data, "delay")
    _validate_nonnegative_integer(period, "period")
    return data.shift(periods=period)


def delta(data: PandasObject, period: int = 1) -> PandasObject:
    """当前值减去同一股票 period 个观测前的值。"""
    _validate_time_input(data, "delta")
    _validate_nonnegative_integer(period, "period")
    return data.diff(periods=period)


def rolling_mean(data: PandasObject, window: int) -> PandasObject:
    """每只股票过去 window 个观测的滚动均值。"""
    _validate_time_input(data, "rolling_mean")
    _validate_positive_integer(window, "window")
    return data.rolling(window=window, min_periods=window).mean()


def rolling_sum(data: PandasObject, window: int) -> PandasObject:
    """每只股票过去 window 个观测的滚动和。"""
    _validate_time_input(data, "rolling_sum")
    _validate_positive_integer(window, "window")
    return data.rolling(window=window, min_periods=window).sum()


def rolling_std(
    data: PandasObject,
    window: int,
    ddof: int = 1,
) -> PandasObject:
    """每只股票过去 window 个观测的滚动标准差。"""
    _validate_time_input(data, "rolling_std")
    _validate_positive_integer(window, "window")
    _validate_nonnegative_integer(ddof, "ddof")
    return data.rolling(window=window, min_periods=window).std(ddof=ddof)


def rolling_correlation(
    left: PandasObject,
    right: PandasObject,
    window: int,
) -> PandasObject:
    """逐股票计算两个变量过去 window 个观测的滚动相关系数。"""
    _validate_pair(left, right, "rolling_correlation")
    _validate_positive_integer(window, "window")
    result = left.rolling(window=window, min_periods=window).corr(right)
    # 零方差窗口的相关系数未定义，不应保留为无穷值。
    return result.replace([np.inf, -np.inf], np.nan)


def rolling_covariance(
    left: PandasObject,
    right: PandasObject,
    window: int,
    ddof: int = 1,
) -> PandasObject:
    """逐股票计算两个变量过去 window 个观测的滚动协方差。"""
    _validate_pair(left, right, "rolling_covariance")
    _validate_positive_integer(window, "window")
    _validate_nonnegative_integer(ddof, "ddof")
    return left.rolling(window=window, min_periods=window).cov(
        right,
        ddof=ddof,
    )


def ts_min(data: PandasObject, window: int) -> PandasObject:
    """每只股票过去 window 个观测的最小值。"""
    _validate_time_input(data, "ts_min")
    _validate_positive_integer(window, "window")
    return data.rolling(window=window, min_periods=window).min()


def ts_max(data: PandasObject, window: int) -> PandasObject:
    """每只股票过去 window 个观测的最大值。"""
    _validate_time_input(data, "ts_max")
    _validate_positive_integer(window, "window")
    return data.rolling(window=window, min_periods=window).max()


def ts_argmax(data: PandasObject, window: int) -> PandasObject:
    """
    返回窗口最大值的位置。

    位置从窗口最早观测开始计数：1 表示最早，window 表示当前观测。
    若最大值并列，返回最早出现的位置。
    """
    _validate_time_input(data, "ts_argmax")
    _validate_positive_integer(window, "window")
    return data.rolling(window=window, min_periods=window).apply(
        lambda values: float(np.argmax(values) + 1),
        raw=True,
    )


def ts_argmin(data: PandasObject, window: int) -> PandasObject:
    """
    返回窗口最小值的位置。

    位置从窗口最早观测开始计数：1 表示最早，window 表示当前观测。
    若最小值并列，返回最早出现的位置。
    """
    _validate_time_input(data, "ts_argmin")
    _validate_positive_integer(window, "window")
    return data.rolling(window=window, min_periods=window).apply(
        lambda values: float(np.argmin(values) + 1),
        raw=True,
    )


def scale(frame: pd.DataFrame, k: float = 1.0) -> pd.DataFrame:
    """
    每个交易日进行横截面缩放，使 sum(abs(x)) 等于 k。

    某日全部有效值均为 0 时无法缩放，该行返回 NaN。
    """
    _validate_panel(frame, "scale")
    if not np.isfinite(k) or k <= 0:
        raise ValueError("k 必须是大于 0 的有限数值")
    prepared = _apply_universe_mask(frame, "scale")
    denominator = prepared.abs().sum(axis=1, min_count=1).replace(0, np.nan)
    return prepared.mul(k).div(denominator, axis=0)


def sign(data: PandasObject) -> PandasObject:
    """正数返回 1，负数返回 -1，0 返回 0，缺失值保持缺失。"""
    _validate_time_input(data, "sign")
    return np.sign(data)


def abs(data: PandasObject) -> PandasObject:
    """逐元素取绝对值。"""
    _validate_time_input(data, "abs")
    return data.abs()


def decay_linear(data: PandasObject, window: int) -> PandasObject:
    """线性衰减加权均值；越接近当前日期的观测权重越大。"""
    _validate_time_input(data, "decay_linear")
    _validate_positive_integer(window, "window")
    weights = np.arange(1.0, window + 1.0)
    weights /= weights.sum()
    return data.rolling(window=window, min_periods=window).apply(
        lambda values: float(np.dot(values, weights)),
        raw=True,
    )


def _validate_elementwise_inputs(
    left: PandasObject,
    right: PandasObject | float,
    name: str,
) -> None:
    _validate_time_input(left, name)
    if isinstance(right, (pd.Series, pd.DataFrame)):
        _validate_pair(left, right, name)
    elif not np.isscalar(right):
        raise TypeError(f"{name}的right必须是标量或与left对齐的pandas对象")


def elementwise_max(
    left: PandasObject,
    right: PandasObject | float,
) -> PandasObject:
    """两个对齐表达式逐元素取最大值。"""
    _validate_elementwise_inputs(left, right, "elementwise_max")
    return np.maximum(left, right)


def elementwise_min(
    left: PandasObject,
    right: PandasObject | float,
) -> PandasObject:
    """两个对齐表达式逐元素取最小值。"""
    _validate_elementwise_inputs(left, right, "elementwise_min")
    return np.minimum(left, right)


def industry_neutralize(
    data: pd.DataFrame,
    groups: pd.DataFrame | pd.Series,
) -> pd.DataFrame:
    """逐日按行业组减去组内均值。

    groups 可以是与 data 同形状的动态分类宽表，也可以是以股票代码为
    index 的静态分类 Series。分类缺失或原值缺失的位置保持 NaN。
    """
    _validate_panel(data, "industry_neutralize")
    if isinstance(groups, pd.DataFrame):
        if not groups.index.equals(data.index):
            raise ValueError("industry_neutralize的行业index与数据不一致")
        if not groups.columns.equals(data.columns):
            raise ValueError("industry_neutralize的行业columns与数据不一致")
    elif isinstance(groups, pd.Series):
        if not groups.index.equals(data.columns):
            raise ValueError("industry_neutralize的静态行业index必须等于股票列")
    else:
        raise TypeError("industry_neutralize的groups必须是DataFrame或Series")

    prepared = _apply_universe_mask(data, "industry_neutralize")
    values = prepared.to_numpy(dtype=float, na_value=np.nan)
    dynamic_labels = (
        groups.to_numpy(dtype=object)
        if isinstance(groups, pd.DataFrame)
        else None
    )
    static_labels = (
        groups.to_numpy(dtype=object)
        if isinstance(groups, pd.Series)
        else None
    )
    result = np.full(values.shape, np.nan, dtype=float)

    for row_number in range(values.shape[0]):
        row_values = values[row_number]
        row_labels = (
            dynamic_labels[row_number]
            if dynamic_labels is not None
            else static_labels
        )
        valid = ~np.isnan(row_values) & pd.notna(row_labels)
        if not valid.any():
            continue
        group_codes, _ = pd.factorize(row_labels[valid], sort=False)
        group_sums = np.bincount(group_codes, weights=row_values[valid])
        group_counts = np.bincount(group_codes)
        group_means = group_sums / group_counts
        result[row_number, valid] = (
            row_values[valid] - group_means[group_codes]
        )

    return pd.DataFrame(result, index=data.index, columns=data.columns)


def log(data: PandasObject) -> PandasObject:
    """逐元素自然对数；非正数超出定义域并返回NaN。"""
    _validate_time_input(data, "log")
    return np.log(data.where(data > 0))


def rolling_product(data: PandasObject, window: int) -> PandasObject:
    """每只股票过去window个观测的滚动连乘。"""
    _validate_time_input(data, "rolling_product")
    _validate_positive_integer(window, "window")
    return data.rolling(window=window, min_periods=window).apply(
        np.prod,
        raw=True,
    )


def signed_power(
    data: PandasObject,
    exponent: PandasObject | float,
) -> PandasObject:
    """保留底数符号的幂：sign(x) * abs(x) ** exponent。"""
    _validate_time_input(data, "signed_power")
    if isinstance(exponent, (pd.Series, pd.DataFrame)):
        _validate_pair(data, exponent, "signed_power")
    elif not np.isscalar(exponent):
        raise TypeError("signed_power的exponent必须是标量或对齐的pandas对象")

    result = np.sign(data) * data.abs().pow(exponent)
    return result.replace([np.inf, -np.inf], np.nan)
