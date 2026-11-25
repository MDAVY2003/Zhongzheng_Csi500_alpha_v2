"""统一管理研究项目的预热、样本内和样本外日期边界。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class SamplePeriod:
    """一个闭区间样本阶段。"""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("样本阶段名称不能为空")
        if self.start > self.end:
            raise ValueError(f"{self.name}的开始日期不能晚于结束日期")

    def select(self, frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """按日期闭区间截取对象，不修改传入数据。"""
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError("待切分对象必须使用DatetimeIndex")
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError("待切分对象的日期必须唯一并按升序排列")
        selected = frame.loc[(frame.index >= self.start) & (frame.index <= self.end)]
        if selected.empty:
            raise ValueError(
                f"数据不包含{self.name}区间："
                f"{self.start.date()}至{self.end.date()}"
            )
        return selected.copy(deep=False)


WARMUP_PERIOD = SamplePeriod(
    name="warmup",
    start=pd.Timestamp("2017-01-01"),
    end=pd.Timestamp("2017-12-31"),
)
IN_SAMPLE_PERIOD = SamplePeriod(
    name="in_sample",
    start=pd.Timestamp("2018-01-01"),
    end=pd.Timestamp("2022-12-31"),
)
OUT_OF_SAMPLE_PERIOD = SamplePeriod(
    name="out_of_sample",
    start=pd.Timestamp("2023-01-01"),
    end=pd.Timestamp("2025-12-31"),
)


def validate_sample_periods() -> None:
    """确认三个阶段互不重叠且按时间顺序排列。"""
    periods = (WARMUP_PERIOD, IN_SAMPLE_PERIOD, OUT_OF_SAMPLE_PERIOD)
    for left, right in zip(periods, periods[1:]):
        if left.end >= right.start:
            raise ValueError(f"{left.name}与{right.name}存在重叠")


validate_sample_periods()
