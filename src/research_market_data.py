from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

def _pivot(
    frame: pd.DataFrame,
    value: str,
    index: pd.DatetimeIndex,
    columns: pd.Index,
    *,
    dtype: str = "float64",
) -> pd.DataFrame:
    return (
        frame.pivot(index="date", columns="code", values=value)
        .reindex(index=index, columns=columns)
        .astype(dtype)
    )

def expand_full_day_suspensions(
    events: pd.DataFrame,
    index: pd.DatetimeIndex,
    columns: pd.Index,
) -> pd.DataFrame:
    """把S/R事件展开成逐日全日停牌状态；缺失开盘价仍独立阻止成交。"""
    panel = pd.DataFrame(False, index=index, columns=columns)
    events = events.sort_values(["code", "date", "suspend_type"])
    grouped = {code: group for code, group in events.groupby("code", sort=False)}
    for code in columns:
        group = grouped.get(code)
        if group is None:
            continue
        state_changes: dict[pd.Timestamp, bool] = {}
        for date, day_events in group.groupby("date", sort=False):
            has_full_day_s = bool(
                (
                    day_events["suspend_type"].eq("S")
                    & day_events["suspend_timing"].isna()
                ).any()
            )
            if has_full_day_s:
                state_changes[pd.Timestamp(date)] = True
            elif day_events["suspend_type"].eq("R").any():
                state_changes[pd.Timestamp(date)] = False
        state = pd.Series(state_changes, dtype="boolean")
        panel[code] = state.reindex(index).ffill().fillna(False).astype(bool)
    return panel

def load_market_data(
    *,
    raw_daily_dir: Path,
    membership_file: Path,
    market_cap_file: Path,
    limits_file: Path,
    suspensions_file: Path,
    reference_factor: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """加载股票市场面板，并从固定锚点前复权行情反推未复权开盘价。"""
    index = reference_factor.index
    columns = reference_factor.columns
    frames: list[pd.DataFrame] = []
    for code in tqdm(columns, desc="读取股票行情", unit="stock"):
        path = raw_daily_dir / f"{code}.parquet"
        if not path.exists():
            # Keep this historical member as an all-NaN panel column. It is not
            # silently removed from universe counts or candidate coverage.
            continue
        frame = pd.read_parquet(
            path,
            columns=[
                "date", "code", "open", "close", "preclose", "amount",
                "adj_factor", "adjust_type",
            ],
        )
        if frame.empty:
            continue
        if not frame["adjust_type"].eq("qfq").all():
            raise ValueError(f"{code}行情不是统一qfq口径")
        final_factor = pd.to_numeric(frame["adj_factor"], errors="coerce").dropna()
        if final_factor.empty or final_factor.iloc[-1] <= 0:
            raise ValueError(f"{code}缺少有效adj_factor")
        anchor = float(final_factor.iloc[-1])
        frame["raw_open"] = frame["open"] * anchor / frame["adj_factor"]
        frame["raw_close"] = frame["close"] * anchor / frame["adj_factor"]
        frame["raw_preclose_reconstructed"] = (
            frame["preclose"] * anchor / frame["adj_factor"]
        )
        frames.append(
            frame[
                [
                    "date",
                    "code",
                    "open",
                    "close",
                    "raw_open",
                    "raw_close",
                    "raw_preclose_reconstructed",
                    "amount",
                ]
            ]
        )
    long_market = pd.concat(frames, ignore_index=True)
    long_market["date"] = pd.to_datetime(long_market["date"], errors="raise")
    if long_market.duplicated(["date", "code"]).any():
        raise ValueError("股票行情存在重复date-code")

    membership = pd.read_parquet(membership_file)
    membership["date"] = pd.to_datetime(membership["date"], errors="raise")
    membership = membership.loc[membership["is_member"], ["date", "code", "is_member"]]
    if membership.duplicated(["date", "code"]).any():
        raise ValueError("历史股票池存在重复date-code")
    membership_panel = _pivot(
        membership, "is_member", index, columns, dtype="boolean"
    ).fillna(False).astype(bool)

    limits = pd.read_parquet(limits_file)
    limits["date"] = pd.to_datetime(limits["date"], errors="raise")
    if limits.duplicated(["date", "code"]).any():
        raise ValueError("涨跌停数据存在重复date-code")

    market_cap = pd.read_parquet(
        market_cap_file, columns=["date", "code", "circulating_market_cap"]
    )
    market_cap["date"] = pd.to_datetime(market_cap["date"], errors="raise")
    if market_cap.duplicated(["date", "code"]).any():
        raise ValueError("流通市值存在重复date-code")

    suspensions = pd.read_parquet(suspensions_file)
    suspensions["date"] = pd.to_datetime(suspensions["date"], errors="raise")
    suspension_panel = expand_full_day_suspensions(suspensions, index, columns)

    panels = {
        "qfq_open": _pivot(long_market, "open", index, columns),
        "qfq_close": _pivot(long_market, "close", index, columns),
        "raw_open": _pivot(long_market, "raw_open", index, columns),
        "raw_close": _pivot(long_market, "raw_close", index, columns),
        "amount": _pivot(long_market, "amount", index, columns),
        "circulating_market_cap": _pivot(
            market_cap, "circulating_market_cap", index, columns
        ),
        "up_limit": _pivot(limits, "up_limit", index, columns),
        "down_limit": _pivot(limits, "down_limit", index, columns),
        "full_day_suspension": suspension_panel,
        "membership": membership_panel,
    }

    comparison = long_market.merge(
        limits[["date", "code", "pre_close"]],
        on=["date", "code"],
        how="inner",
    )
    comparison["preclose_relative_error"] = (
        comparison["raw_preclose_reconstructed"] - comparison["pre_close"]
    ).abs().div(comparison["pre_close"].where(comparison["pre_close"].gt(0)))
    valid_error = comparison["preclose_relative_error"].dropna()
    diagnostics = pd.DataFrame(
        [
            {
                "check": "raw_price_reconstruction",
                "status": "pass" if valid_error.quantile(0.99) < 0.005 else "failed",
                "rows": len(valid_error),
                "value": float(valid_error.quantile(0.99)),
                "threshold": 0.005,
                "message": "反推未复权preclose与交易所pre_close的99%分位相对误差",
            },
            {
                "check": "historical_membership_count",
                "status": "pass",
                "rows": int(membership_panel.sum().sum()),
                "value": float(membership_panel.sum(axis=1).median()),
                "threshold": 500,
                "message": "逐日历史中证500成员数量中位数",
            },
        ]
    )
    if diagnostics["status"].eq("failed").any():
        raise ValueError(diagnostics.loc[diagnostics["status"].eq("failed"), "message"].iloc[0])
    return panels, diagnostics

def load_futures(futures_dir: Path) -> dict[str, pd.DataFrame]:
    datasets = {
        "basic": pd.read_parquet(futures_dir / "fut_basic.parquet"),
        "daily": pd.read_parquet(futures_dir / "fut_daily.parquet"),
        "settle": pd.read_parquet(futures_dir / "fut_settle.parquet"),
    }
    if datasets["daily"].duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("IC日线存在重复trade_date-ts_code")
    if datasets["settle"].duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("IC结算参数存在重复trade_date-ts_code")
    return datasets
