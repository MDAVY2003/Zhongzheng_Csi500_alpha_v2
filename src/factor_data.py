"""把市值和申万行业成员区间转换为Alpha计算需要的宽表。"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping

import pandas as pd


INDUSTRY_REQUIRED_COLUMNS = {
    "code",
    "sector_code",
    "industry_code",
    "subindustry_code",
    "in_date",
    "out_date",
}


@dataclass(frozen=True, slots=True)
class FactorData:
    """一个 Alpha 所需的 date x code 宽表集合。

    所有已提供字段必须使用相同的交易日索引和股票代码列。
    未被当前因子使用的字段可以不提供。
    """

    open: pd.DataFrame | None = None
    high: pd.DataFrame | None = None
    low: pd.DataFrame | None = None
    close: pd.DataFrame | None = None
    volume: pd.DataFrame | None = None
    amount: pd.DataFrame | None = None
    vwap: pd.DataFrame | None = None
    returns: pd.DataFrame | None = None
    cap: pd.DataFrame | None = None
    sector: pd.DataFrame | None = None
    industry: pd.DataFrame | None = None
    subindustry: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        panels = self.as_dict()
        if not panels:
            raise ValueError("FactorData至少需要一个宽表字段")

        reference_name, reference = next(iter(panels.items()))
        for name, panel in panels.items():
            if not isinstance(panel, pd.DataFrame):
                raise TypeError(f"FactorData.{name}必须是pandas DataFrame")
            if not isinstance(panel.index, pd.DatetimeIndex):
                raise TypeError(f"FactorData.{name}必须使用DatetimeIndex")
            if not panel.index.is_unique or not panel.index.is_monotonic_increasing:
                raise ValueError(f"FactorData.{name}的日期必须唯一并按升序排列")
            if not panel.columns.is_unique:
                raise ValueError(f"FactorData.{name}的股票代码列不能重复")
            if not panel.index.equals(reference.index):
                raise ValueError(
                    f"FactorData.{name}与{reference_name}的日期索引不一致"
                )
            if not panel.columns.equals(reference.columns):
                raise ValueError(
                    f"FactorData.{name}与{reference_name}的股票列不一致"
                )

    @classmethod
    def from_mapping(cls, panels: Mapping[str, pd.DataFrame]) -> "FactorData":
        """从字段名到宽表的映射创建 FactorData。"""
        valid_names = {item.name for item in fields(cls)}
        unknown = set(panels).difference(valid_names)
        if unknown:
            raise ValueError(f"FactorData出现未知字段：{sorted(unknown)}")
        return cls(**dict(panels))

    @property
    def available_fields(self) -> tuple[str, ...]:
        """返回当前已提供的字段名。"""
        return tuple(self.as_dict())

    @property
    def index(self) -> pd.DatetimeIndex:
        """返回统一的交易日索引。"""
        return next(iter(self.as_dict().values())).index

    @property
    def columns(self) -> pd.Index:
        """返回统一的股票代码列。"""
        return next(iter(self.as_dict().values())).columns

    def as_dict(self) -> dict[str, pd.DataFrame]:
        """返回所有已提供字段，不复制底层 DataFrame。"""
        return {
            item.name: value
            for item in fields(self)
            if (value := getattr(self, item.name)) is not None
        }

    def require(self, *field_names: str) -> tuple[pd.DataFrame, ...]:
        """取出必需字段；字段未提供时立即报错。"""
        missing = [name for name in field_names if getattr(self, name, None) is None]
        if missing:
            raise ValueError(f"FactorData缺少字段：{missing}")
        return tuple(getattr(self, name) for name in field_names)


def load_cap_panel(
    file_path: Path,
    *,
    dates: pd.DatetimeIndex | None = None,
    codes: list[str] | None = None,
) -> pd.DataFrame:
    """读取日度cap长表并转换为date x code宽表。"""
    data = pd.read_parquet(file_path, columns=["date", "code", "cap"])
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    if data.duplicated(["date", "code"]).any():
        raise ValueError("日度市值存在重复date-code记录")
    panel = data.pivot(index="date", columns="code", values="cap").sort_index()
    if dates is not None:
        panel = panel.reindex(index=dates)
    if codes is not None:
        panel = panel.reindex(columns=codes)
    return panel


def build_sw_industry_panels(
    membership: pd.DataFrame,
    *,
    dates: pd.DatetimeIndex,
    codes: list[str],
) -> dict[str, pd.DataFrame]:
    """按成员区间展开申万一、二、三级动态分类宽表。

    in_date和out_date均按包含边界处理；同一日期存在重叠记录时，
    in_date较晚的记录后写入并优先。
    """
    missing = INDUSTRY_REQUIRED_COLUMNS.difference(membership.columns)
    if missing:
        raise ValueError(f"申万行业成员数据缺少字段：{sorted(missing)}")
    if not isinstance(dates, pd.DatetimeIndex):
        raise TypeError("dates必须是DatetimeIndex")
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("dates必须唯一并按升序排列")
    if len(codes) != len(set(codes)):
        raise ValueError("codes不能重复")

    data = membership.copy()
    data["code"] = data["code"].astype("string").str.upper()
    data["in_date"] = pd.to_datetime(data["in_date"], errors="coerce")
    data["out_date"] = pd.to_datetime(data["out_date"], errors="coerce")
    data = data.dropna(subset=["code", "in_date"]).sort_values(
        ["code", "in_date"]
    )

    level_columns = {
        "sector": "sector_code",
        "industry": "industry_code",
        "subindustry": "subindustry_code",
    }
    panels = {
        level: pd.DataFrame(pd.NA, index=dates, columns=codes, dtype="string")
        for level in level_columns
    }

    for code, records in data[data["code"].isin(codes)].groupby(
        "code",
        sort=False,
    ):
        for record in records.itertuples(index=False):
            start = record.in_date
            end = record.out_date
            active = dates >= start
            if pd.notna(end):
                active &= dates <= end
            active_dates = dates[active]
            for level, column in level_columns.items():
                value = getattr(record, column)
                if pd.notna(value):
                    panels[level].loc[active_dates, code] = value
    return panels


def load_sw_industry_panels(
    file_path: Path,
    *,
    dates: pd.DatetimeIndex,
    codes: list[str],
) -> dict[str, pd.DataFrame]:
    """读取申万成员区间文件并展开为动态分类宽表。"""
    membership = pd.read_parquet(file_path)
    return build_sw_industry_panels(
        membership,
        dates=dates,
        codes=codes,
    )
