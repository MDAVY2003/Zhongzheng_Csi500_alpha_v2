"""WorldQuant 101 Alphas 第一、二批公式实现。

所有因子使用 ``alphaXXX(data: FactorData)`` 统一接口，输出为
``date x code`` 宽表。公式来自项目指定的 ``101Alpha_code_1.py``
中 Alpha#1...Alpha#101 公式注释。

小数窗口向下取整；advN 使用人民币成交金额的滚动均值。
rank 与 ts_rank 均使用百分位尺度。修正版本：formula-v2。
条件公式使用三值条件语义；行业公式使用按日期对齐的历史
申万一、二、三级分类。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import operators as op
from factor_data import FactorData


IMPLEMENTED_ALPHA_IDS = tuple(range(1, 102))

__all__ = [
    *(f"alpha{alpha_id:03d}" for alpha_id in IMPLEMENTED_ALPHA_IDS),
    "alpha_012",
    "alpha_033",
    "alpha_041",
    "alpha_042",
    "alpha_101",
]


def _window(value: float) -> int:
    """按原论文将小数窗口向下取整为正整数。"""
    window = int(np.floor(float(value)))
    if window <= 0:
        raise ValueError("窗口向下取整后必须大于0")
    return window


def _adv(data: FactorData, window: float) -> pd.DataFrame:
    """advN：过去N日平均成交金额（CNY），不使用成交股数。"""
    (amount,) = data.require("amount")
    return op.rolling_mean(amount, _window(window))


def _safe_divide(
    numerator: pd.DataFrame,
    denominator: pd.DataFrame,
) -> pd.DataFrame:
    """仅将除数为0的未定义结果保留为NaN。"""
    result = numerator / denominator.mask(denominator == 0)
    return result.replace([np.inf, -np.inf], np.nan)


def _comparison(
    left: pd.DataFrame | float,
    right: pd.DataFrame | float,
    operator: str,
) -> pd.DataFrame:
    """逐元素比较并保留任一输入的缺失位置。"""
    reference = left if isinstance(left, pd.DataFrame) else right
    if not isinstance(reference, pd.DataFrame):
        raise TypeError("比较至少需要一个DataFrame输入")
    if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        if not left.index.equals(right.index) or not left.columns.equals(right.columns):
            raise ValueError("比较的两个DataFrame必须完全对齐")

    operations = {
        "lt": lambda: left < right,
        "le": lambda: left <= right,
        "eq": lambda: left == right,
    }
    if operator not in operations:
        raise ValueError(f"未知比较算子：{operator}")
    result = operations[operator]().astype(float)
    missing = pd.DataFrame(False, index=reference.index, columns=reference.columns)
    if isinstance(left, pd.DataFrame):
        missing |= left.isna()
    if isinstance(right, pd.DataFrame):
        missing |= right.isna()
    return result.mask(missing)


def _less_than(
    left: pd.DataFrame | float,
    right: pd.DataFrame | float,
) -> pd.DataFrame:
    return _comparison(left, right, "lt")


def _less_equal(
    left: pd.DataFrame | float,
    right: pd.DataFrame | float,
) -> pd.DataFrame:
    return _comparison(left, right, "le")


def _equal(
    left: pd.DataFrame | float,
    right: pd.DataFrame | float,
) -> pd.DataFrame:
    return _comparison(left, right, "eq")


def _logical_or(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """对0/1/NaN条件执行三值逻辑或。"""
    if not left.index.equals(right.index) or not left.columns.equals(right.columns):
        raise ValueError("逻辑或的两个条件必须完全对齐")
    result = pd.DataFrame(np.nan, index=left.index, columns=left.columns)
    result[(left == 1) | (right == 1)] = 1.0
    result[(left == 0) & (right == 0)] = 0.0
    return result


def _as_panel(
    value: pd.DataFrame | float,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        if not value.index.equals(reference.index) or not value.columns.equals(
            reference.columns
        ):
            raise ValueError("条件分支DataFrame必须完全对齐")
        return value
    return pd.DataFrame(float(value), index=reference.index, columns=reference.columns)


def _conditional(
    condition: pd.DataFrame,
    value_if_true: pd.DataFrame | float,
    value_if_false: pd.DataFrame | float,
) -> pd.DataFrame:
    """三值条件选择；条件缺失时输出保持NaN。"""
    true_panel = _as_panel(value_if_true, condition)
    false_panel = _as_panel(value_if_false, condition)
    result = false_panel.where(condition == 0)
    return result.where(condition != 1, true_panel)


# 第一批：直接量价公式


def alpha005(data: FactorData) -> pd.DataFrame:
    """Alpha 005."""
    open_price, close, vwap = data.require("open", "close", "vwap")
    first = op.rank(open_price - op.rolling_sum(vwap, 10) / 10)
    second = -op.abs(op.rank(close - vwap))
    return first * second


def alpha008(data: FactorData) -> pd.DataFrame:
    """Alpha 008."""
    open_price, returns = data.require("open", "returns")
    product = op.rolling_sum(open_price, 5) * op.rolling_sum(returns, 5)
    return -op.rank(product - op.delay(product, 10))


def alpha011(data: FactorData) -> pd.DataFrame:
    """Alpha 011."""
    close, volume, vwap = data.require("close", "volume", "vwap")
    spread = vwap - close
    return (
        op.rank(op.ts_max(spread, 3)) + op.rank(op.ts_min(spread, 3))
    ) * op.rank(op.delta(volume, 3))


def alpha012(data: FactorData) -> pd.DataFrame:
    """Alpha 012."""
    close, volume = data.require("close", "volume")
    return op.sign(op.delta(volume, 1)) * (-op.delta(close, 1))


def alpha019(data: FactorData) -> pd.DataFrame:
    """Alpha 019."""
    close, returns = data.require("close", "returns")
    direction = -op.sign((close - op.delay(close, 7)) + op.delta(close, 7))
    return direction * (1 + op.rank(1 + op.rolling_sum(returns, 250)))


def alpha020(data: FactorData) -> pd.DataFrame:
    """Alpha 020."""
    open_price, high, low, close = data.require("open", "high", "low", "close")
    return (
        -op.rank(open_price - op.delay(high, 1))
        * op.rank(open_price - op.delay(close, 1))
        * op.rank(open_price - op.delay(low, 1))
    )


def alpha025(data: FactorData) -> pd.DataFrame:
    """Alpha 025."""
    high, close, vwap, returns, volume = data.require(
        "high", "close", "vwap", "returns", "volume"
    )
    return op.rank((-returns) * _adv(data, 20) * vwap * (high - close))


def alpha030(data: FactorData) -> pd.DataFrame:
    """Alpha 030."""
    close, volume = data.require("close", "volume")
    direction = (
        op.sign(close - op.delay(close, 1))
        + op.sign(op.delay(close, 1) - op.delay(close, 2))
        + op.sign(op.delay(close, 2) - op.delay(close, 3))
    )
    numerator = (1.0 - op.rank(direction)) * op.rolling_sum(volume, 5)
    return _safe_divide(numerator, op.rolling_sum(volume, 20))


def alpha033(data: FactorData) -> pd.DataFrame:
    """Alpha 033."""
    open_price, close = data.require("open", "close")
    return op.rank(-(1 - _safe_divide(open_price, close)).pow(1))


def alpha034(data: FactorData) -> pd.DataFrame:
    """Alpha 034."""
    close, returns = data.require("close", "returns")
    volatility_ratio = _safe_divide(
        op.rolling_std(returns, 2),
        op.rolling_std(returns, 5),
    )
    return op.rank(
        (1 - op.rank(volatility_ratio)) + (1 - op.rank(op.delta(close, 1)))
    )


def alpha041(data: FactorData) -> pd.DataFrame:
    """Alpha 041."""
    high, low, vwap = data.require("high", "low", "vwap")
    return (high * low).pow(0.5) - vwap


def alpha042(data: FactorData) -> pd.DataFrame:
    """Alpha 042."""
    close, vwap = data.require("close", "vwap")
    return _safe_divide(op.rank(vwap - close), op.rank(vwap + close))


def alpha047(data: FactorData) -> pd.DataFrame:
    """Alpha 047."""
    high, close, volume, vwap = data.require("high", "close", "volume", "vwap")
    first = _safe_divide(op.rank(1 / close) * volume, _adv(data, 20))
    second = _safe_divide(
        high * op.rank(high - close),
        op.rolling_sum(high, 5) / 5,
    )
    return first * second - op.rank(vwap - op.delay(vwap, 5))


def alpha053(data: FactorData) -> pd.DataFrame:
    """Alpha 053."""
    high, low, close = data.require("high", "low", "close")
    inner = _safe_divide((close - low) - (high - close), close - low)
    return -op.delta(inner, 9)


def alpha054(data: FactorData) -> pd.DataFrame:
    """Alpha 054."""
    open_price, high, low, close = data.require("open", "high", "low", "close")
    numerator = -(low - close) * open_price.pow(5)
    denominator = (low - high) * close.pow(5)
    return _safe_divide(numerator, denominator)


def alpha060(data: FactorData) -> pd.DataFrame:
    """Alpha 060."""
    high, low, close, volume = data.require("high", "low", "close", "volume")
    position = _safe_divide((close - low) - (high - close), high - low) * volume
    return -(2 * op.scale(op.rank(position)) - op.scale(op.rank(op.ts_argmax(close, 10))))


def alpha083(data: FactorData) -> pd.DataFrame:
    """Alpha 083."""
    high, low, close, volume, vwap = data.require(
        "high", "low", "close", "volume", "vwap"
    )
    range_to_mean = _safe_divide(high - low, op.rolling_sum(close, 5) / 5)
    numerator = op.rank(op.delay(range_to_mean, 2)) * op.rank(op.rank(volume))
    denominator = _safe_divide(range_to_mean, vwap - close)
    return _safe_divide(numerator, denominator)


def alpha101(data: FactorData) -> pd.DataFrame:
    """Alpha 101."""
    open_price, high, low, close = data.require("open", "high", "low", "close")
    return (close - open_price) / ((high - low) + 0.001)


# 第二批：复杂时间序列公式


def alpha002(data: FactorData) -> pd.DataFrame:
    """Alpha 002."""
    open_price, close, volume = data.require("open", "close", "volume")
    left = op.rank(op.delta(op.log(volume), 2))
    right = op.rank(_safe_divide(close - open_price, open_price))
    return -op.rolling_correlation(left, right, 6)


def alpha003(data: FactorData) -> pd.DataFrame:
    """Alpha 003."""
    open_price, volume = data.require("open", "volume")
    return -op.rolling_correlation(op.rank(open_price), op.rank(volume), 10)


def alpha004(data: FactorData) -> pd.DataFrame:
    """Alpha 004."""
    (low,) = data.require("low")
    return -op.ts_rank(op.rank(low), 9)


def alpha006(data: FactorData) -> pd.DataFrame:
    """Alpha 006."""
    open_price, volume = data.require("open", "volume")
    return -op.rolling_correlation(open_price, volume, 10)


def alpha013(data: FactorData) -> pd.DataFrame:
    """Alpha 013."""
    close, volume = data.require("close", "volume")
    covariance = op.rolling_covariance(op.rank(close), op.rank(volume), 5)
    return -op.rank(covariance)


def alpha014(data: FactorData) -> pd.DataFrame:
    """Alpha 014."""
    open_price, volume, returns = data.require("open", "volume", "returns")
    return -op.rank(op.delta(returns, 3)) * op.rolling_correlation(
        open_price, volume, 10
    )


def alpha015(data: FactorData) -> pd.DataFrame:
    """Alpha 015."""
    high, volume = data.require("high", "volume")
    correlation = op.rolling_correlation(op.rank(high), op.rank(volume), 3)
    return -op.rolling_sum(op.rank(correlation), 3)


def alpha016(data: FactorData) -> pd.DataFrame:
    """Alpha 016."""
    high, volume = data.require("high", "volume")
    covariance = op.rolling_covariance(op.rank(high), op.rank(volume), 5)
    return -op.rank(covariance)


def alpha017(data: FactorData) -> pd.DataFrame:
    """Alpha 017."""
    close, volume = data.require("close", "volume")
    return (
        -op.rank(op.ts_rank(close, 10))
        * op.rank(op.delta(op.delta(close, 1), 1))
        * op.rank(op.ts_rank(_safe_divide(volume, _adv(data, 20)), 5))
    )


def alpha018(data: FactorData) -> pd.DataFrame:
    """Alpha 018."""
    open_price, close = data.require("open", "close")
    inner = (
        op.rolling_std(op.abs(close - open_price), 5)
        + (close - open_price)
        + op.rolling_correlation(close, open_price, 10)
    )
    return -op.rank(inner)


def alpha022(data: FactorData) -> pd.DataFrame:
    """Alpha 022."""
    high, close, volume = data.require("high", "close", "volume")
    correlation_change = op.delta(op.rolling_correlation(high, volume, 5), 5)
    return -(correlation_change * op.rank(op.rolling_std(close, 20)))


def alpha026(data: FactorData) -> pd.DataFrame:
    """Alpha 026."""
    high, volume = data.require("high", "volume")
    correlation = op.rolling_correlation(
        op.ts_rank(volume, 5), op.ts_rank(high, 5), 5
    )
    return -op.ts_max(correlation, 3)


def alpha028(data: FactorData) -> pd.DataFrame:
    """Alpha 028."""
    high, low, close, volume = data.require("high", "low", "close", "volume")
    correlation = op.rolling_correlation(_adv(data, 20), low, 5)
    return op.scale(correlation + (high + low) / 2 - close)


def alpha029(data: FactorData) -> pd.DataFrame:
    """Alpha 029."""
    close, returns = data.require("close", "returns")
    nested = -op.rank(op.delta(close - 1, 5))
    nested = op.rank(op.rank(nested))
    nested = op.ts_min(nested, 2)
    nested = op.rolling_sum(nested, 1)
    nested = op.log(nested)
    nested = op.scale(nested)
    nested = op.rank(op.rank(nested))
    nested = op.rolling_product(nested, 1)
    first = op.ts_min(nested, 5)
    second = op.ts_rank(op.delay(-returns, 6), 5)
    return first + second


def alpha031(data: FactorData) -> pd.DataFrame:
    """Alpha 031."""
    low, close, volume = data.require("low", "close", "volume")
    first = op.decay_linear(-op.rank(op.rank(op.delta(close, 10))), 10)
    first = op.rank(op.rank(op.rank(first)))
    second = op.rank(-op.delta(close, 3))
    third = op.sign(op.scale(op.rolling_correlation(_adv(data, 20), low, 12)))
    return first + second + third


def alpha032(data: FactorData) -> pd.DataFrame:
    """Alpha 032."""
    close, vwap = data.require("close", "vwap")
    first = op.scale(op.rolling_sum(close, 7) / 7 - close)
    second = 20 * op.scale(
        op.rolling_correlation(vwap, op.delay(close, 5), 230)
    )
    return first + second


def alpha035(data: FactorData) -> pd.DataFrame:
    """Alpha 035."""
    high, low, close, volume, returns = data.require(
        "high", "low", "close", "volume", "returns"
    )
    return (
        op.ts_rank(volume, 32)
        * (1 - op.ts_rank(close + high - low, 16))
        * (1 - op.ts_rank(returns, 32))
    )


def alpha036(data: FactorData) -> pd.DataFrame:
    """Alpha 036."""
    open_price, close, volume, vwap, returns = data.require(
        "open", "close", "volume", "vwap", "returns"
    )
    return (
        2.21
        * op.rank(
            op.rolling_correlation(close - open_price, op.delay(volume, 1), 15)
        )
        + 0.7 * op.rank(open_price - close)
        + 0.73 * op.rank(op.ts_rank(op.delay(-returns, 6), 5))
        + op.rank(op.abs(op.rolling_correlation(vwap, _adv(data, 20), 6)))
        + 0.6
        * op.rank(
            (op.rolling_sum(close, 200) / 200 - open_price)
            * (close - open_price)
        )
    )


def alpha037(data: FactorData) -> pd.DataFrame:
    """Alpha 037."""
    open_price, close = data.require("open", "close")
    return op.rank(
        op.rolling_correlation(op.delay(open_price - close, 1), close, 200)
    ) + op.rank(open_price - close)


def alpha038(data: FactorData) -> pd.DataFrame:
    """Alpha 038."""
    open_price, close = data.require("open", "close")
    return -op.rank(op.ts_rank(close, 10)) * op.rank(
        _safe_divide(close, open_price)
    )


def alpha039(data: FactorData) -> pd.DataFrame:
    """Alpha 039."""
    close, volume, returns = data.require("close", "volume", "returns")
    decayed = op.decay_linear(_safe_divide(volume, _adv(data, 20)), 9)
    return -op.rank(op.delta(close, 7) * (1 - op.rank(decayed))) * (
        1 + op.rank(op.rolling_sum(returns, 250))
    )


def alpha040(data: FactorData) -> pd.DataFrame:
    """Alpha 040."""
    high, volume = data.require("high", "volume")
    return -op.rank(op.rolling_std(high, 10)) * op.rolling_correlation(
        high, volume, 10
    )


def alpha043(data: FactorData) -> pd.DataFrame:
    """Alpha 043."""
    close, volume = data.require("close", "volume")
    return op.ts_rank(_safe_divide(volume, _adv(data, 20)), 20) * op.ts_rank(
        -op.delta(close, 7), 8
    )


def alpha044(data: FactorData) -> pd.DataFrame:
    """Alpha 044."""
    high, volume = data.require("high", "volume")
    return -op.rolling_correlation(high, op.rank(volume), 5)


def alpha045(data: FactorData) -> pd.DataFrame:
    """Alpha 045."""
    close, volume = data.require("close", "volume")
    first = op.rank(op.rolling_sum(op.delay(close, 5), 20) / 20)
    second = op.rolling_correlation(close, volume, 2)
    third = op.rank(
        op.rolling_correlation(
            op.rolling_sum(close, 5), op.rolling_sum(close, 20), 2
        )
    )
    return -(first * second * third)


def alpha050(data: FactorData) -> pd.DataFrame:
    """Alpha 050."""
    volume, vwap = data.require("volume", "vwap")
    correlation = op.rolling_correlation(op.rank(volume), op.rank(vwap), 5)
    return -op.ts_max(op.rank(correlation), 5)


def alpha052(data: FactorData) -> pd.DataFrame:
    """Alpha 052."""
    low, volume, returns = data.require("low", "volume", "returns")
    low_min = op.ts_min(low, 5)
    first = -low_min + op.delay(low_min, 5)
    second = op.rank(
        (op.rolling_sum(returns, 240) - op.rolling_sum(returns, 20)) / 220
    )
    return first * second * op.ts_rank(volume, 5)


def alpha055(data: FactorData) -> pd.DataFrame:
    """Alpha 055."""
    high, low, close, volume = data.require("high", "low", "close", "volume")
    position = _safe_divide(
        close - op.ts_min(low, 12),
        op.ts_max(high, 12) - op.ts_min(low, 12),
    )
    return -op.rolling_correlation(op.rank(position), op.rank(volume), 6)


def alpha057(data: FactorData) -> pd.DataFrame:
    """Alpha 057."""
    close, vwap = data.require("close", "vwap")
    denominator = op.decay_linear(op.rank(op.ts_argmax(close, 30)), 2)
    return -_safe_divide(close - vwap, denominator)


def alpha061(data: FactorData) -> pd.DataFrame:
    """Alpha 061."""
    volume, vwap = data.require("volume", "vwap")
    left = op.rank(vwap - op.ts_min(vwap, _window(16.1219)))
    right = op.rank(
        op.rolling_correlation(vwap, _adv(data, 180), _window(17.9282))
    )
    return _less_than(left, right)


def alpha062(data: FactorData) -> pd.DataFrame:
    """Alpha 062."""
    open_price, high, low, volume, vwap = data.require(
        "open", "high", "low", "volume", "vwap"
    )
    left = op.rank(
        op.rolling_correlation(
            vwap,
            op.rolling_sum(_adv(data, 20), _window(22.4101)),
            _window(9.91009),
        )
    )
    inner = _less_than(
        op.rank(open_price) + op.rank(open_price),
        op.rank((high + low) / 2) + op.rank(high),
    )
    return -_less_than(left, op.rank(inner))


def alpha064(data: FactorData) -> pd.DataFrame:
    """Alpha 064."""
    open_price, high, low, volume, vwap = data.require(
        "open", "high", "low", "volume", "vwap"
    )
    weighted_price = open_price * 0.178404 + low * (1 - 0.178404)
    left = op.rank(
        op.rolling_correlation(
            op.rolling_sum(weighted_price, _window(12.7054)),
            op.rolling_sum(_adv(data, 120), _window(12.7054)),
            _window(16.6208),
        )
    )
    right_value = (high + low) / 2 * 0.178404 + vwap * (1 - 0.178404)
    right = op.rank(op.delta(right_value, _window(3.69741)))
    return -_less_than(left, right)


def alpha065(data: FactorData) -> pd.DataFrame:
    """Alpha 065."""
    open_price, volume, vwap = data.require("open", "volume", "vwap")
    weighted_price = open_price * 0.00817205 + vwap * (1 - 0.00817205)
    left = op.rank(
        op.rolling_correlation(
            weighted_price,
            op.rolling_sum(_adv(data, 60), _window(8.6911)),
            _window(6.40374),
        )
    )
    right = op.rank(open_price - op.ts_min(open_price, _window(13.635)))
    return -_less_than(left, right)


def alpha066(data: FactorData) -> pd.DataFrame:
    """Alpha 066."""
    open_price, high, low, vwap = data.require("open", "high", "low", "vwap")
    first = op.rank(
        op.decay_linear(
            op.delta(vwap, _window(3.51013)),
            _window(7.23052),
        )
    )
    weighted_low = low * 0.96633 + low * (1 - 0.96633)
    ratio = _safe_divide(
        weighted_low - vwap,
        open_price - (high + low) / 2,
    )
    second = op.ts_rank(
        op.decay_linear(ratio, _window(11.4157)),
        _window(6.72611),
    )
    return -(first + second)


def alpha068(data: FactorData) -> pd.DataFrame:
    """Alpha 068."""
    high, low, close, volume = data.require("high", "low", "close", "volume")
    left = op.ts_rank(
        op.rolling_correlation(
            op.rank(high), op.rank(_adv(data, 15)), _window(8.91644)
        ),
        _window(13.9333),
    )
    right = op.rank(
        op.delta(
            close * 0.518371 + low * (1 - 0.518371),
            _window(1.06157),
        )
    )
    return -_less_than(left, right)


def alpha071(data: FactorData) -> pd.DataFrame:
    """Alpha 071."""
    open_price, low, close, volume, vwap = data.require(
        "open", "low", "close", "volume", "vwap"
    )
    first = op.ts_rank(
        op.decay_linear(
            op.rolling_correlation(
                op.ts_rank(close, _window(3.43976)),
                op.ts_rank(_adv(data, 180), _window(12.0647)),
                _window(18.0175),
            ),
            _window(4.20501),
        ),
        _window(15.6948),
    )
    second = op.ts_rank(
        op.decay_linear(
            op.rank((low + open_price) - (vwap + vwap)).pow(2),
            _window(16.4662),
        ),
        _window(4.4388),
    )
    return op.elementwise_max(first, second)


def alpha072(data: FactorData) -> pd.DataFrame:
    """Alpha 072."""
    high, low, volume, vwap = data.require("high", "low", "volume", "vwap")
    first = op.rank(
        op.decay_linear(
            op.rolling_correlation(
                (high + low) / 2,
                _adv(data, 40),
                _window(8.93345),
            ),
            _window(10.1519),
        )
    )
    second = op.rank(
        op.decay_linear(
            op.rolling_correlation(
                op.ts_rank(vwap, _window(3.72469)),
                op.ts_rank(volume, _window(18.5188)),
                _window(6.86671),
            ),
            _window(2.95011),
        )
    )
    return _safe_divide(first, second)


def alpha073(data: FactorData) -> pd.DataFrame:
    """Alpha 073."""
    open_price, low, vwap = data.require("open", "low", "vwap")
    first = op.rank(
        op.decay_linear(
            op.delta(vwap, _window(4.72775)),
            _window(2.91864),
        )
    )
    weighted_price = open_price * 0.147155 + low * (1 - 0.147155)
    relative_change = -_safe_divide(
        op.delta(weighted_price, _window(2.03608)), weighted_price
    )
    second = op.ts_rank(
        op.decay_linear(relative_change, _window(3.33829)),
        _window(16.7411),
    )
    return -op.elementwise_max(first, second)


def alpha074(data: FactorData) -> pd.DataFrame:
    """Alpha 074."""
    high, close, volume, vwap = data.require("high", "close", "volume", "vwap")
    left = op.rank(
        op.rolling_correlation(
            close,
            op.rolling_sum(_adv(data, 30), _window(37.4843)),
            _window(15.1365),
        )
    )
    weighted_price = high * 0.0261661 + vwap * (1 - 0.0261661)
    right = op.rank(
        op.rolling_correlation(
            op.rank(weighted_price),
            op.rank(volume),
            _window(11.4791),
        )
    )
    return -_less_than(left, right)


def alpha075(data: FactorData) -> pd.DataFrame:
    """Alpha 075."""
    low, volume, vwap = data.require("low", "volume", "vwap")
    left = op.rank(
        op.rolling_correlation(vwap, volume, _window(4.24304))
    )
    right = op.rank(
        op.rolling_correlation(
            op.rank(low), op.rank(_adv(data, 50)), _window(12.4413)
        )
    )
    return _less_than(left, right)


def alpha077(data: FactorData) -> pd.DataFrame:
    """Alpha 077."""
    high, low, volume, vwap = data.require("high", "low", "volume", "vwap")
    first = op.rank(
        op.decay_linear(
            ((high + low) / 2 + high) - (vwap + high),
            _window(20.0451),
        )
    )
    second = op.rank(
        op.decay_linear(
            op.rolling_correlation(
                (high + low) / 2,
                _adv(data, 40),
                _window(3.1614),
            ),
            _window(5.64125),
        )
    )
    return op.elementwise_min(first, second)


def alpha078(data: FactorData) -> pd.DataFrame:
    """Alpha 078."""
    low, volume, vwap = data.require("low", "volume", "vwap")
    weighted_price = low * 0.352233 + vwap * (1 - 0.352233)
    base = op.rank(
        op.rolling_correlation(
            op.rolling_sum(weighted_price, _window(19.7428)),
            op.rolling_sum(_adv(data, 40), _window(19.7428)),
            _window(6.83313),
        )
    )
    exponent = op.rank(
        op.rolling_correlation(
            op.rank(vwap), op.rank(volume), _window(5.77492)
        )
    )
    return base.pow(exponent)


def alpha081(data: FactorData) -> pd.DataFrame:
    """Alpha 081."""
    volume, vwap = data.require("volume", "vwap")
    correlation = op.rolling_correlation(
        vwap,
        op.rolling_sum(_adv(data, 10), _window(49.6054)),
        _window(8.47743),
    )
    powered = op.rank(correlation).pow(4)
    left = op.rank(
        op.log(
            op.rolling_product(
                op.rank(powered),
                _window(14.9655),
            )
        )
    )
    right = op.rank(
        op.rolling_correlation(
            op.rank(vwap), op.rank(volume), _window(5.07914)
        )
    )
    return -_less_than(left, right)


def alpha084(data: FactorData) -> pd.DataFrame:
    """Alpha 084."""
    close, vwap = data.require("close", "vwap")
    base = op.ts_rank(
        vwap - op.ts_max(vwap, _window(15.3217)),
        _window(20.7127),
    )
    exponent = op.delta(close, _window(4.96796))
    return op.signed_power(base, exponent)


def alpha085(data: FactorData) -> pd.DataFrame:
    """Alpha 085."""
    high, low, close, volume = data.require("high", "low", "close", "volume")
    base = op.rank(
        op.rolling_correlation(
            high * 0.876703 + close * (1 - 0.876703),
            _adv(data, 30),
            _window(9.61331),
        )
    )
    exponent = op.rank(
        op.rolling_correlation(
            op.ts_rank((high + low) / 2, _window(3.70596)),
            op.ts_rank(volume, _window(10.1595)),
            _window(7.11408),
        )
    )
    return base.pow(exponent)


def alpha086(data: FactorData) -> pd.DataFrame:
    """Alpha 086."""
    open_price, close, volume, vwap = data.require(
        "open", "close", "volume", "vwap"
    )
    left = op.ts_rank(
        op.rolling_correlation(
            close,
            op.rolling_sum(_adv(data, 20), _window(14.7444)),
            _window(6.00049),
        ),
        _window(20.4195),
    )
    right = op.rank((open_price + close) - (vwap + open_price))
    return -_less_than(left, right)


def alpha088(data: FactorData) -> pd.DataFrame:
    """Alpha 088."""
    open_price, high, low, close, volume = data.require(
        "open", "high", "low", "close", "volume"
    )
    first = op.rank(
        op.decay_linear(
            (op.rank(open_price) + op.rank(low))
            - (op.rank(high) + op.rank(close)),
            _window(8.06882),
        )
    )
    second = op.ts_rank(
        op.decay_linear(
            op.rolling_correlation(
                op.ts_rank(close, _window(8.44728)),
                op.ts_rank(_adv(data, 60), _window(20.6966)),
                _window(8.01266),
            ),
            _window(6.65053),
        ),
        _window(2.61957),
    )
    return op.elementwise_min(first, second)


def alpha092(data: FactorData) -> pd.DataFrame:
    """Alpha 092."""
    open_price, high, low, close, volume = data.require(
        "open", "high", "low", "close", "volume"
    )
    condition = _less_than((high + low) / 2 + close, low + open_price)
    first = op.ts_rank(
        op.decay_linear(condition, _window(14.7221)),
        _window(18.8683),
    )
    second = op.ts_rank(
        op.decay_linear(
            op.rolling_correlation(
                op.rank(low),
                op.rank(_adv(data, 30)),
                _window(7.58555),
            ),
            _window(6.94024),
        ),
        _window(6.80584),
    )
    return op.elementwise_min(first, second)


def alpha094(data: FactorData) -> pd.DataFrame:
    """Alpha 094."""
    volume, vwap = data.require("volume", "vwap")
    base = op.rank(vwap - op.ts_min(vwap, _window(11.5783)))
    exponent = op.ts_rank(
        op.rolling_correlation(
            op.ts_rank(vwap, _window(19.6462)),
            op.ts_rank(_adv(data, 60), _window(4.02992)),
            _window(18.0926),
        ),
        _window(2.70756),
    )
    return -base.pow(exponent)


def alpha095(data: FactorData) -> pd.DataFrame:
    """Alpha 095."""
    open_price, high, low, volume = data.require("open", "high", "low", "volume")
    left = op.rank(open_price - op.ts_min(open_price, _window(12.4105)))
    correlation = op.rolling_correlation(
        op.rolling_sum((high + low) / 2, _window(19.1351)),
        op.rolling_sum(_adv(data, 40), _window(19.1351)),
        _window(12.8742),
    )
    right = op.ts_rank(op.rank(correlation).pow(5), _window(11.7584))
    return _less_than(left, right)


def alpha096(data: FactorData) -> pd.DataFrame:
    """Alpha 096."""
    close, volume, vwap = data.require("close", "volume", "vwap")
    first = op.ts_rank(
        op.decay_linear(
            op.rolling_correlation(
                op.rank(vwap), op.rank(volume), _window(3.83878)
            ),
            _window(4.16783),
        ),
        _window(8.38151),
    )
    inner = op.rolling_correlation(
        op.ts_rank(close, _window(7.45404)),
        op.ts_rank(_adv(data, 60), _window(4.13242)),
        _window(3.65459),
    )
    second = op.ts_rank(
        op.decay_linear(
            op.ts_argmax(inner, _window(12.6556)),
            _window(14.0365),
        ),
        _window(13.4143),
    )
    return -op.elementwise_max(first, second)


def alpha098(data: FactorData) -> pd.DataFrame:
    """Alpha 098."""
    open_price, volume, vwap = data.require("open", "volume", "vwap")
    first = op.rank(
        op.decay_linear(
            op.rolling_correlation(
                vwap,
                op.rolling_sum(_adv(data, 5), _window(26.4719)),
                _window(4.58418),
            ),
            _window(7.18088),
        )
    )
    inner = op.rolling_correlation(
        op.rank(open_price),
        op.rank(_adv(data, 15)),
        _window(20.8187),
    )
    second = op.rank(
        op.decay_linear(
            op.ts_rank(
                op.ts_argmin(inner, _window(8.62571)),
                _window(6.95668),
            ),
            _window(8.07206),
        )
    )
    return first - second


def alpha099(data: FactorData) -> pd.DataFrame:
    """Alpha 099."""
    high, low, volume = data.require("high", "low", "volume")
    left = op.rank(
        op.rolling_correlation(
            op.rolling_sum((high + low) / 2, _window(19.8975)),
            op.rolling_sum(_adv(data, 60), _window(19.8975)),
            _window(8.8136),
        )
    )
    right = op.rank(
        op.rolling_correlation(low, volume, _window(6.28259))
    )
    return -_less_than(left, right)


# 第三批：条件表达式公式


def alpha001(data: FactorData) -> pd.DataFrame:
    """Alpha 001."""
    close, returns = data.require("close", "returns")
    condition = _less_than(returns, 0.0)
    selected = _conditional(
        condition,
        op.rolling_std(returns, 20),
        close,
    )
    return op.rank(op.ts_argmax(op.signed_power(selected, 2.0), 5)) - 0.5


def alpha007(data: FactorData) -> pd.DataFrame:
    """Alpha 007."""
    close, volume = data.require("close", "volume")
    change = op.delta(close, 7)
    signal = -op.ts_rank(op.abs(change), 60) * op.sign(change)
    return _conditional(_less_than(_adv(data, 20), volume), signal, -1.0)


def alpha009(data: FactorData) -> pd.DataFrame:
    """Alpha 009."""
    (close,) = data.require("close")
    change = op.delta(close, 1)
    first = _less_than(0.0, op.ts_min(change, 5))
    second = _less_than(op.ts_max(change, 5), 0.0)
    return _conditional(first, change, _conditional(second, change, -change))


def alpha010(data: FactorData) -> pd.DataFrame:
    """Alpha 010."""
    (close,) = data.require("close")
    change = op.delta(close, 1)
    first = _less_than(0.0, op.ts_min(change, 4))
    second = _less_than(op.ts_max(change, 4), 0.0)
    selected = _conditional(first, change, _conditional(second, change, -change))
    return op.rank(selected)


def alpha021(data: FactorData) -> pd.DataFrame:
    """Alpha 021."""
    close, volume = data.require("close", "volume")
    mean_8 = op.rolling_sum(close, 8) / 8
    mean_2 = op.rolling_sum(close, 2) / 2
    std_8 = op.rolling_std(close, 8)
    first = _less_than(mean_8 + std_8, mean_2)
    second = _less_than(mean_2, mean_8 - std_8)
    volume_ratio = _safe_divide(volume, _adv(data, 20))
    third = _logical_or(
        _less_than(1.0, volume_ratio),
        _equal(volume_ratio, 1.0),
    )
    return _conditional(
        first,
        -1.0,
        _conditional(second, 1.0, _conditional(third, 1.0, -1.0)),
    )


def alpha023(data: FactorData) -> pd.DataFrame:
    """Alpha 023."""
    (high,) = data.require("high")
    condition = _less_than(op.rolling_sum(high, 20) / 20, high)
    return _conditional(condition, -op.delta(high, 2), 0.0)


def alpha024(data: FactorData) -> pd.DataFrame:
    """Alpha 024."""
    (close,) = data.require("close")
    mean_100 = op.rolling_sum(close, 100) / 100
    ratio = _safe_divide(op.delta(mean_100, 100), op.delay(close, 100))
    condition = _less_equal(ratio, 0.05)
    return _conditional(
        condition,
        -(close - op.ts_min(close, 100)),
        -op.delta(close, 3),
    )


def alpha027(data: FactorData) -> pd.DataFrame:
    """Alpha 027."""
    volume, vwap = data.require("volume", "vwap")
    correlation = op.rolling_correlation(op.rank(volume), op.rank(vwap), 6)
    signal = op.rank(op.rolling_sum(correlation, 2) / 2.0)
    return _conditional(_less_than(0.5, signal), -1.0, 1.0)


def _alpha046_trend(close: pd.DataFrame) -> pd.DataFrame:
    return (
        (op.delay(close, 20) - op.delay(close, 10)) / 10
        - (op.delay(close, 10) - close) / 10
    )


def alpha046(data: FactorData) -> pd.DataFrame:
    """Alpha 046."""
    (close,) = data.require("close")
    trend = _alpha046_trend(close)
    return _conditional(
        _less_than(0.25, trend),
        -1.0,
        _conditional(_less_than(trend, 0.0), 1.0, -op.delta(close, 1)),
    )


def alpha049(data: FactorData) -> pd.DataFrame:
    """Alpha 049."""
    (close,) = data.require("close")
    trend = _alpha046_trend(close)
    return _conditional(
        _less_than(trend, -0.1),
        1.0,
        -op.delta(close, 1),
    )


def alpha051(data: FactorData) -> pd.DataFrame:
    """Alpha 051."""
    (close,) = data.require("close")
    trend = _alpha046_trend(close)
    return _conditional(
        _less_than(trend, -0.05),
        1.0,
        -op.delta(close, 1),
    )


# 第四批：行业中性化和市值依赖公式


def alpha048(data: FactorData) -> pd.DataFrame:
    """Alpha 048，申万三级行业中性化。"""
    close, subindustry = data.require("close", "subindustry")
    change = op.delta(close, 1)
    lagged_change = op.delta(op.delay(close, 1), 1)
    numerator = _safe_divide(
        op.rolling_correlation(change, lagged_change, 250) * change,
        close,
    )
    numerator = op.industry_neutralize(numerator, subindustry)
    relative_change = _safe_divide(change, op.delay(close, 1)).pow(2)
    denominator = op.rolling_sum(relative_change, 250)
    return _safe_divide(numerator, denominator)


def alpha056(data: FactorData) -> pd.DataFrame:
    """Alpha 056，使用每日总市值。"""
    returns, cap = data.require("returns", "cap")
    ratio = _safe_divide(
        op.rolling_sum(returns, 10),
        op.rolling_sum(op.rolling_sum(returns, 2), 3),
    )
    return -(op.rank(ratio) * op.rank(returns * cap))


def alpha058(data: FactorData) -> pd.DataFrame:
    """Alpha 058，申万一级行业中性化。"""
    volume, vwap, sector = data.require("volume", "vwap", "sector")
    neutral_vwap = op.industry_neutralize(vwap, sector)
    correlation = op.rolling_correlation(neutral_vwap, volume, _window(3.92795))
    return -op.ts_rank(
        op.decay_linear(correlation, _window(7.89291)),
        _window(5.50322),
    )


def alpha059(data: FactorData) -> pd.DataFrame:
    """Alpha 059，申万二级行业中性化。"""
    volume, vwap, industry = data.require("volume", "vwap", "industry")
    weighted_vwap = vwap * 0.728317 + vwap * (1 - 0.728317)
    neutral_vwap = op.industry_neutralize(weighted_vwap, industry)
    correlation = op.rolling_correlation(neutral_vwap, volume, _window(4.25197))
    return -op.ts_rank(
        op.decay_linear(correlation, _window(16.2289)),
        _window(8.19648),
    )


def alpha063(data: FactorData) -> pd.DataFrame:
    """Alpha 063，申万二级行业中性化。"""
    open_price, close, volume, vwap, industry = data.require(
        "open", "close", "volume", "vwap", "industry"
    )
    neutral_close = op.industry_neutralize(close, industry)
    first = op.rank(
        op.decay_linear(
            op.delta(neutral_close, _window(2.25164)),
            _window(8.22237),
        )
    )
    weighted_price = vwap * 0.318108 + open_price * (1 - 0.318108)
    second = op.rank(
        op.decay_linear(
            op.rolling_correlation(
                weighted_price,
                op.rolling_sum(_adv(data, 180), _window(37.2467)),
                _window(13.557),
            ),
            _window(12.2883),
        )
    )
    return -(first - second)


def alpha067(data: FactorData) -> pd.DataFrame:
    """Alpha 067，同时使用申万一级和三级行业。"""
    high, volume, vwap, sector, subindustry = data.require(
        "high", "volume", "vwap", "sector", "subindustry"
    )
    base = op.rank(high - op.ts_min(high, _window(2.14593)))
    exponent = op.rank(
        op.rolling_correlation(
            op.industry_neutralize(vwap, sector),
            op.industry_neutralize(_adv(data, 20), subindustry),
            _window(6.02936),
        )
    )
    return -base.pow(exponent)


def alpha069(data: FactorData) -> pd.DataFrame:
    """Alpha 069，申万二级行业中性化。"""
    close, volume, vwap, industry = data.require(
        "close", "volume", "vwap", "industry"
    )
    neutral_vwap = op.industry_neutralize(vwap, industry)
    base = op.rank(
        op.ts_max(
            op.delta(neutral_vwap, _window(2.72412)),
            _window(4.79344),
        )
    )
    weighted_price = close * 0.490655 + vwap * (1 - 0.490655)
    exponent = op.ts_rank(
        op.rolling_correlation(
            weighted_price, _adv(data, 20), _window(4.92416)
        ),
        _window(9.0615),
    )
    return -base.pow(exponent)


def alpha070(data: FactorData) -> pd.DataFrame:
    """Alpha 070，申万二级行业中性化。"""
    close, volume, vwap, industry = data.require(
        "close", "volume", "vwap", "industry"
    )
    base = op.rank(op.delta(vwap, _window(1.29456)))
    exponent = op.ts_rank(
        op.rolling_correlation(
            op.industry_neutralize(close, industry),
            _adv(data, 50),
            _window(17.8256),
        ),
        _window(17.9171),
    )
    return -base.pow(exponent)


def alpha076(data: FactorData) -> pd.DataFrame:
    """Alpha 076，申万一级行业中性化。"""
    low, volume, vwap, sector = data.require("low", "volume", "vwap", "sector")
    first = op.rank(
        op.decay_linear(
            op.delta(vwap, _window(1.24383)),
            _window(11.8259),
        )
    )
    correlation = op.rolling_correlation(
        op.industry_neutralize(low, sector),
        _adv(data, 81),
        _window(8.14941),
    )
    second = op.ts_rank(
        op.decay_linear(
            op.ts_rank(correlation, _window(19.569)),
            _window(17.1543),
        ),
        _window(19.383),
    )
    return -op.elementwise_max(first, second)


def alpha079(data: FactorData) -> pd.DataFrame:
    """Alpha 079，申万一级行业中性化。"""
    open_price, close, volume, vwap, sector = data.require(
        "open", "close", "volume", "vwap", "sector"
    )
    weighted_price = close * 0.60733 + open_price * (1 - 0.60733)
    left = op.rank(
        op.delta(
            op.industry_neutralize(weighted_price, sector),
            _window(1.23438),
        )
    )
    right = op.rank(
        op.rolling_correlation(
            op.ts_rank(vwap, _window(3.60973)),
            op.ts_rank(_adv(data, 150), _window(9.18637)),
            _window(14.6644),
        )
    )
    return _less_than(left, right)


def alpha080(data: FactorData) -> pd.DataFrame:
    """Alpha 080，申万二级行业中性化。"""
    open_price, high, volume, industry = data.require(
        "open", "high", "volume", "industry"
    )
    weighted_price = open_price * 0.868128 + high * (1 - 0.868128)
    base = op.rank(
        op.sign(
            op.delta(
                op.industry_neutralize(weighted_price, industry),
                _window(4.04545),
            )
        )
    )
    exponent = op.ts_rank(
        op.rolling_correlation(high, _adv(data, 10), _window(5.11456)),
        _window(5.53756),
    )
    return -base.pow(exponent)


def alpha082(data: FactorData) -> pd.DataFrame:
    """Alpha 082，申万一级行业中性化。"""
    open_price, volume, sector = data.require("open", "volume", "sector")
    first = op.rank(
        op.decay_linear(
            op.delta(open_price, _window(1.46063)),
            _window(14.8717),
        )
    )
    weighted_open = open_price * 0.634196 + open_price * (1 - 0.634196)
    second = op.ts_rank(
        op.decay_linear(
            op.rolling_correlation(
                op.industry_neutralize(volume, sector),
                weighted_open,
                _window(17.4842),
            ),
            _window(6.92131),
        ),
        _window(13.4283),
    )
    return -op.elementwise_min(first, second)


def alpha087(data: FactorData) -> pd.DataFrame:
    """Alpha 087，申万二级行业中性化。"""
    close, volume, vwap, industry = data.require(
        "close", "volume", "vwap", "industry"
    )
    weighted_price = close * 0.369701 + vwap * (1 - 0.369701)
    first = op.rank(
        op.decay_linear(
            op.delta(weighted_price, _window(1.91233)),
            _window(2.65461),
        )
    )
    correlation = op.rolling_correlation(
        op.industry_neutralize(_adv(data, 81), industry),
        close,
        _window(13.4132),
    )
    second = op.ts_rank(
        op.decay_linear(op.abs(correlation), _window(4.89768)),
        _window(14.4535),
    )
    return -op.elementwise_max(first, second)


def alpha089(data: FactorData) -> pd.DataFrame:
    """Alpha 089，申万二级行业中性化。"""
    low, volume, vwap, industry = data.require("low", "volume", "vwap", "industry")
    weighted_low = low * 0.967285 + low * (1 - 0.967285)
    first = op.ts_rank(
        op.decay_linear(
            op.rolling_correlation(
                weighted_low, _adv(data, 10), _window(6.94279)
            ),
            _window(5.51607),
        ),
        _window(3.79744),
    )
    second = op.ts_rank(
        op.decay_linear(
            op.delta(
                op.industry_neutralize(vwap, industry),
                _window(3.48158),
            ),
            _window(10.1466),
        ),
        _window(15.3012),
    )
    return first - second


def alpha090(data: FactorData) -> pd.DataFrame:
    """Alpha 090，申万三级行业中性化。"""
    low, close, volume, subindustry = data.require(
        "low", "close", "volume", "subindustry"
    )
    base = op.rank(close - op.ts_max(close, _window(4.66719)))
    exponent = op.ts_rank(
        op.rolling_correlation(
            op.industry_neutralize(_adv(data, 40), subindustry),
            low,
            _window(5.38375),
        ),
        _window(3.21856),
    )
    return -base.pow(exponent)


def alpha091(data: FactorData) -> pd.DataFrame:
    """Alpha 091，申万二级行业中性化。"""
    close, volume, vwap, industry = data.require(
        "close", "volume", "vwap", "industry"
    )
    correlation = op.rolling_correlation(
        op.industry_neutralize(close, industry),
        volume,
        _window(9.74928),
    )
    first = op.ts_rank(
        op.decay_linear(
            op.decay_linear(correlation, _window(16.398)),
            _window(3.83219),
        ),
        _window(4.8667),
    )
    second = op.rank(
        op.decay_linear(
            op.rolling_correlation(vwap, _adv(data, 30), _window(4.01303)),
            _window(2.6809),
        )
    )
    return -(first - second)


def alpha093(data: FactorData) -> pd.DataFrame:
    """Alpha 093，申万二级行业中性化。"""
    close, volume, vwap, industry = data.require(
        "close", "volume", "vwap", "industry"
    )
    numerator = op.ts_rank(
        op.decay_linear(
            op.rolling_correlation(
                op.industry_neutralize(vwap, industry),
                _adv(data, 81),
                _window(17.4193),
            ),
            _window(19.848),
        ),
        _window(7.54455),
    )
    weighted_price = close * 0.524434 + vwap * (1 - 0.524434)
    denominator = op.rank(
        op.decay_linear(
            op.delta(weighted_price, _window(2.77377)),
            _window(16.2664),
        )
    )
    return _safe_divide(numerator, denominator)


def alpha097(data: FactorData) -> pd.DataFrame:
    """Alpha 097，申万二级行业中性化。"""
    low, volume, vwap, industry = data.require("low", "volume", "vwap", "industry")
    weighted_price = low * 0.721001 + vwap * (1 - 0.721001)
    first = op.rank(
        op.decay_linear(
            op.delta(
                op.industry_neutralize(weighted_price, industry),
                _window(3.3705),
            ),
            _window(20.4523),
        )
    )
    inner = op.ts_rank(
        op.rolling_correlation(
            op.ts_rank(low, _window(7.87871)),
            op.ts_rank(_adv(data, 60), _window(17.255)),
            _window(4.97547),
        ),
        _window(18.5925),
    )
    second = op.ts_rank(
        op.decay_linear(inner, _window(15.7152)),
        _window(6.71659),
    )
    return -(first - second)


def alpha100(data: FactorData) -> pd.DataFrame:
    """Alpha 100，申万三级行业中性化。"""
    high, low, close, volume, subindustry = data.require(
        "high", "low", "close", "volume", "subindustry"
    )
    adv20 = _adv(data, 20)
    position = _safe_divide((close - low) - (high - close), high - low) * volume
    first = op.rank(position)
    first = op.industry_neutralize(first, subindustry)
    first = op.industry_neutralize(first, subindustry)
    first = 1.5 * op.scale(first)

    second = (
        op.rolling_correlation(close, op.rank(adv20), 5)
        - op.rank(op.ts_argmin(close, 30))
    )
    second = op.scale(op.industry_neutralize(second, subindustry))
    return -((first - second) * _safe_divide(volume, adv20))


# 保留旧的多参数入口，便于已有教程和测试继续运行。


def alpha_012(volume: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    return alpha012(FactorData(volume=volume, close=close))


def alpha_033(open_price: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    return alpha033(FactorData(open=open_price, close=close))


def alpha_041(
    high: pd.DataFrame,
    low: pd.DataFrame,
    vwap: pd.DataFrame,
) -> pd.DataFrame:
    return alpha041(FactorData(high=high, low=low, vwap=vwap))


def alpha_042(vwap: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    return alpha042(FactorData(vwap=vwap, close=close))


def alpha_101(
    close: pd.DataFrame,
    open_price: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
) -> pd.DataFrame:
    return alpha101(
        FactorData(close=close, open=open_price, high=high, low=low)
    )
