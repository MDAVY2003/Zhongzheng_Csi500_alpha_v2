"""Alpha004、067、088的逐信号日严格点时复权计算。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

import operators as op
from factor_data import FactorData


EXACT_FACTOR_NAMES = ("alpha004", "alpha067", "alpha088")
PRICE_FIELDS = frozenset({"open", "high", "low", "close", "vwap"})
EXACT_CUMULATIVE_LOOKBACK = {
    "alpha004": 9,
    "alpha067": 25,
    "alpha088": 92,
}


def _validate_inputs(
    cumulative_data: FactorData,
    adjustment_factor: pd.DataFrame,
    universe_mask: pd.DataFrame,
) -> None:
    reference = next(iter(cumulative_data.as_dict().values()))
    for name, panel in {
        "adjustment_factor": adjustment_factor,
        "universe_mask": universe_mask,
    }.items():
        if not isinstance(panel, pd.DataFrame):
            raise TypeError(f"{name}必须是DataFrame")
        if not panel.index.equals(reference.index) or not panel.columns.equals(
            reference.columns
        ):
            raise ValueError(f"{name}与行情面板不对齐")


def _anchored_price_window(
    panel: pd.DataFrame,
    adjustment_factor: pd.DataFrame,
    start: int,
    end: int,
) -> pd.DataFrame:
    """把[start, end]历史价格统一复权到信号日end。"""
    anchor = adjustment_factor.iloc[end].replace(0, np.nan)
    return panel.iloc[start : end + 1].div(anchor, axis=1)


def _rank_last(values: pd.DataFrame) -> pd.Series:
    last = values.iloc[-1]
    valid = values.notna().all(axis=0)
    less = values.lt(last, axis=1).sum(axis=0)
    equal = values.eq(last, axis=1).sum(axis=0)
    result = less + (equal + 1) / 2
    return (result.astype(float) / len(values)).where(valid)


def _rank_cross_section(values: pd.Series, member: pd.Series) -> pd.Series:
    return values.where(member).rank(
        method="average",
        pct=True,
        na_option="keep",
    )


def _industry_neutralize_rows(
    values: pd.DataFrame,
    groups: pd.DataFrame,
    mask: pd.DataFrame,
) -> pd.DataFrame:
    """逐行减行业均值；用于严格锚定窗口内的小批量计算。"""
    result = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
    for date in values.index:
        row = values.loc[date]
        labels = groups.loc[date]
        valid = row.notna() & labels.notna() & mask.loc[date]
        if valid.any():
            means = row[valid].groupby(labels[valid]).transform("mean")
            result.loc[date, valid] = row[valid] - means
    return result


def _strict_row_correlation(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> pd.Series:
    """逐股票相关系数；任一窗口缺失或零方差时返回NaN。"""
    valid = left.notna().all(axis=0) & right.notna().all(axis=0)
    left_centered = left - left.mean(axis=0)
    right_centered = right - right.mean(axis=0)
    numerator = (left_centered * right_centered).sum(axis=0)
    denominator = np.sqrt(
        left_centered.pow(2).sum(axis=0)
        * right_centered.pow(2).sum(axis=0)
    )
    return (numerator / denominator.replace(0, np.nan)).where(valid)


def calculate_exact_alpha004(
    cumulative_data: FactorData,
    adjustment_factor: pd.DataFrame,
    universe_mask: pd.DataFrame,
) -> pd.DataFrame:
    """逐日重锚定历史low横截面Rank，再计算9日TS_RANK。"""
    (low,) = cumulative_data.require("low")
    result = pd.DataFrame(np.nan, index=low.index, columns=low.columns)
    for end in range(len(low.index)):
        start = max(0, end - EXACT_CUMULATIVE_LOOKBACK["alpha004"] + 1)
        anchored = _anchored_price_window(low, adjustment_factor, start, end)
        ranked = anchored.where(universe_mask.iloc[start : end + 1]).rank(
            axis=1,
            method="average",
            pct=True,
            na_option="keep",
        )
        if len(ranked) == 9:
            result.iloc[end] = -_rank_last(ranked)
    return result.where(universe_mask)


def calculate_exact_alpha067(
    cumulative_data: FactorData,
    adjustment_factor: pd.DataFrame,
    universe_mask: pd.DataFrame,
) -> pd.DataFrame:
    """逐日重锚定价格，并严格计算行业中性化后的6日相关性。"""
    high, volume, vwap, sector, subindustry = cumulative_data.require(
        "high", "volume", "vwap", "sector", "subindustry"
    )
    result = pd.DataFrame(np.nan, index=high.index, columns=high.columns)
    adv20 = cumulative_data.require("amount")[0].rolling(20, min_periods=20).mean()
    neutral_adv = _industry_neutralize_rows(
        adv20,
        subindustry,
        universe_mask,
    )
    for end in range(len(high.index)):
        if end + 1 < EXACT_CUMULATIVE_LOOKBACK["alpha067"]:
            continue
        high_start = end - 1
        corr_start = end - 5
        member_today = universe_mask.iloc[end]
        anchored_high = _anchored_price_window(
            high, adjustment_factor, high_start, end
        )
        anchored_vwap = _anchored_price_window(
            vwap, adjustment_factor, corr_start, end
        )
        base_input = anchored_high.iloc[-1] - anchored_high.min(axis=0)
        base_input = base_input.where(anchored_high.notna().all(axis=0))
        base = _rank_cross_section(base_input, member_today)
        neutral_vwap = _industry_neutralize_rows(
            anchored_vwap,
            sector.iloc[corr_start : end + 1],
            universe_mask.iloc[corr_start : end + 1],
        )
        correlation = _strict_row_correlation(
            neutral_vwap,
            neutral_adv.iloc[corr_start : end + 1],
        )
        exponent = _rank_cross_section(correlation, member_today)
        result.iloc[end] = -base.pow(exponent)
    return result.where(universe_mask)


def _calculate_alpha088_second_branch(cumulative_data: FactorData) -> pd.DataFrame:
    """该分支仅含逐股票时序运算，可一次向量化且仍为严格结果。"""
    _, _, _, close, volume = cumulative_data.require(
        "open", "high", "low", "close", "volume"
    )
    return op.ts_rank(
        op.decay_linear(
            op.rolling_correlation(
                op.ts_rank(close, 8),
                op.ts_rank(op.rolling_mean(cumulative_data.require("amount")[0], 60), 20),
                8,
            ),
            6,
        ),
        2,
    )


def calculate_exact_alpha088(
    cumulative_data: FactorData,
    adjustment_factor: pd.DataFrame,
    universe_mask: pd.DataFrame,
) -> pd.DataFrame:
    """逐日重锚定四个价格Rank分支，并与严格时序分支合并。"""
    open_price, high, low, close, _ = cumulative_data.require(
        "open", "high", "low", "close", "volume"
    )
    first = pd.DataFrame(np.nan, index=open_price.index, columns=open_price.columns)
    second = _calculate_alpha088_second_branch(cumulative_data)
    for end in range(len(open_price.index)):
        start = max(0, end - 7)
        if end - start + 1 < 8:
            continue
        mask = universe_mask.iloc[start : end + 1]
        anchored = {
            name: _anchored_price_window(
                panel, adjustment_factor, start, end
            )
            for name, panel in {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
            }.items()
        }
        with op.use_universe_mask(mask):
            spread = (
                op.rank(anchored["open"])
                + op.rank(anchored["low"])
                - op.rank(anchored["high"])
                - op.rank(anchored["close"])
            )
            first.iloc[end] = op.rank(op.decay_linear(spread, 8)).iloc[-1]
    result = pd.DataFrame(
        np.minimum(first.to_numpy(), second.to_numpy()),
        index=first.index,
        columns=first.columns,
    )
    return result.where(universe_mask)


_CALCULATORS: dict[
    str,
    Callable[[FactorData, pd.DataFrame, pd.DataFrame], pd.DataFrame],
] = {
    "alpha004": calculate_exact_alpha004,
    "alpha067": calculate_exact_alpha067,
    "alpha088": calculate_exact_alpha088,
}


def calculate_exact_signal_anchor_factors(
    cumulative_data: FactorData,
    adjustment_factor: pd.DataFrame,
    universe_mask: pd.DataFrame,
    *,
    factor_names: tuple[str, ...] = EXACT_FACTOR_NAMES,
) -> dict[str, pd.DataFrame]:
    """计算指定因子并保持完整全市场交易日×股票宽表。"""
    _validate_inputs(cumulative_data, adjustment_factor, universe_mask)
    unknown = set(factor_names).difference(_CALCULATORS)
    if unknown:
        raise ValueError(f"没有严格计算器的因子：{sorted(unknown)}")
    results: dict[str, pd.DataFrame] = {}
    for name in factor_names:
        frame = _CALCULATORS[name](
            cumulative_data,
            adjustment_factor,
            universe_mask,
        ).replace([np.inf, -np.inf], np.nan)
        frame.index.name = "date"
        results[name] = frame
    return results
