"""
WorldQuant 101 学习项目：日频行情预处理层。

本脚本只负责把 data/raw/daily 下的逐股票 Parquet 整理为统一长表，
并输出数据质量报告；不包含 Alpha、IC、RankIC 或回测逻辑。
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


OHLC_COLUMNS = ["open", "high", "low", "close"]
NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
    "returns",
    "pre_close",
    "pct_chg",
]
CORE_VALUE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
    "returns",
    "pre_close",
    "pct_chg",
]
OUTPUT_COLUMNS = [
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
    "returns",
    "pre_close",
    "pct_chg",
    "is_tradeable",
    "has_missing",
    "ohlc_valid",
    "vwap_valid",
    "returns_valid",
    "adjust_type",
    "source_file",
]
RETURN_MISMATCH_TOLERANCE = 1e-4
REPORT_COLUMNS = [
    "source_file",
    "status",
    "rows_raw",
    "rows_output",
    "invalid_key_rows",
    "duplicates_removed",
    "cross_file_duplicates_removed",
    "missing_rows",
    "invalid_ohlc_rows",
    "invalid_vwap_rows",
    "vwap_mismatch_rows",
    "returns_mismatch_rows",
    "untradeable_rows",
    "code_count",
    "codes",
    "start_date",
    "end_date",
    "adjust_types",
    "unit_action",
    "missing_by_column",
    "error_message",
]

OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("date", pa.timestamp("ns")),
        pa.field("code", pa.string()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.float64()),
        pa.field("amount", pa.float64()),
        pa.field("vwap", pa.float64()),
        pa.field("returns", pa.float64()),
        pa.field("pre_close", pa.float64()),
        pa.field("pct_chg", pa.float64()),
        pa.field("is_tradeable", pa.bool_()),
        pa.field("has_missing", pa.bool_()),
        pa.field("ohlc_valid", pa.bool_()),
        pa.field("vwap_valid", pa.bool_()),
        pa.field("returns_valid", pa.bool_()),
        pa.field("adjust_type", pa.string()),
        pa.field("source_file", pa.string()),
    ]
)


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw_daily_dir: Path
    processed_dir: Path
    output_file: Path
    report_file: Path


def get_project_paths() -> ProjectPaths:
    """所有路径均以项目根目录为基准。"""
    root = Path(__file__).resolve().parents[1]
    processed_dir = root / "data" / "processed"
    return ProjectPaths(
        root=root,
        raw_daily_dir=root / "data" / "raw" / "daily",
        processed_dir=processed_dir,
        output_file=processed_dir / "daily_bars.parquet",
        report_file=processed_dir / "data_quality_report.csv",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="合并并检查逐股票日频 Parquet，生成统一长表。"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的 processed 输出文件",
    )
    return parser.parse_args()


def normalize_stock_code(value: Any) -> str | None:
    """统一为 600000.SH、000001.SZ、430047.BJ 形式。"""
    if pd.isna(value):
        return None

    text = str(value).strip().upper()
    suffix_match = re.fullmatch(r"(\d{6})\.(SH|SZ|BJ)", text)
    if suffix_match:
        return f"{suffix_match.group(1)}.{suffix_match.group(2)}"

    prefix_match = re.fullmatch(r"(SH|SZ|BJ)\.(\d{6})", text)
    if prefix_match:
        return f"{prefix_match.group(2)}.{prefix_match.group(1)}"

    if re.fullmatch(r"\d{6}", text):
        if text.startswith(("6", "9")):
            return f"{text}.SH"
        if text.startswith(("0", "3")):
            return f"{text}.SZ"
        if text.startswith(("4", "8")):
            return f"{text}.BJ"

    return None


def empty_series(index: pd.Index) -> pd.Series:
    return pd.Series(pd.NA, index=index, dtype="object")


def first_existing_column(df: pd.DataFrame, names: list[str]) -> pd.Series:
    """按优先级读取第一个存在的字段。"""
    for name in names:
        if name in df.columns:
            return df[name].copy()
    return empty_series(df.index)


def infer_adjust_type(df: pd.DataFrame) -> pd.Series:
    """兼容 Tushare adjust_type 和 BaoStock adjustflag。"""
    if "adjust_type" in df.columns:
        result = df["adjust_type"].astype("string").str.lower()
        return result.fillna("unknown")

    if "adjustflag" in df.columns:
        flags = pd.to_numeric(df["adjustflag"], errors="coerce")
        return flags.map({1: "hfq", 2: "qfq", 3: "none"}).fillna("unknown")

    return pd.Series("unknown", index=df.index, dtype="string")


def prepare_base_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """统一字段名、代码、日期和成交量/成交额单位。"""
    out = pd.DataFrame(index=df.index)
    out["date"] = pd.to_datetime(
        first_existing_column(df, ["date", "trade_date"]),
        errors="coerce",
    ).astype("datetime64[ns]")
    out["code"] = first_existing_column(df, ["code", "ts_code"]).map(
        normalize_stock_code
    ).astype("string")

    for column in OHLC_COLUMNS:
        out[column] = pd.to_numeric(
            first_existing_column(df, [column]),
            errors="coerce",
        )

    out["pre_close"] = pd.to_numeric(
        first_existing_column(df, ["pre_close", "preclose"]),
        errors="coerce",
    )
    out["pct_chg"] = pd.to_numeric(
        first_existing_column(df, ["pct_chg", "pctChg"]),
        errors="coerce",
    )

    # 下载脚本保存的 volume 已是“股”、amount 已是“元”。若直接读到
    # Tushare 原始 vol，则按“手”和“千元”转换。
    if "volume" in df.columns:
        out["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        out["amount"] = pd.to_numeric(
            first_existing_column(df, ["amount"]),
            errors="coerce",
        )
        unit_action = "volume/amount already in shares/yuan"
    elif "vol" in df.columns:
        out["volume"] = pd.to_numeric(df["vol"], errors="coerce") * 100.0
        out["amount"] = pd.to_numeric(
            first_existing_column(df, ["amount"]),
            errors="coerce",
        ) * 1000.0
        unit_action = "converted Tushare vol*100 and amount*1000"
    else:
        out["volume"] = pd.to_numeric(empty_series(df.index), errors="coerce")
        out["amount"] = pd.to_numeric(
            first_existing_column(df, ["amount"]),
            errors="coerce",
        )
        unit_action = "volume missing"

    out["raw_vwap"] = pd.to_numeric(
        first_existing_column(df, ["vwap"]),
        errors="coerce",
    )
    out["raw_returns"] = pd.to_numeric(
        first_existing_column(df, ["returns"]),
        errors="coerce",
    )
    out["trade_status"] = pd.to_numeric(
        first_existing_column(df, ["tradestatus", "trade_status"]),
        errors="coerce",
    )
    out["adjust_type"] = infer_adjust_type(df)
    return out, unit_action


def build_quality_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """计算 OHLC、VWAP、收益率和可交易状态标记。"""
    high_floor = df[["open", "low", "close"]].max(axis=1)
    low_ceiling = df[["open", "high", "close"]].min(axis=1)
    df["ohlc_valid"] = (
        df[OHLC_COLUMNS].notna().all(axis=1)
        & df[OHLC_COLUMNS].gt(0).all(axis=1)
        & df["high"].ge(high_floor)
        & df["low"].le(low_ceiling)
    )

    calculated_vwap = df["amount"].div(df["volume"].where(df["volume"] > 0))
    unadjusted = df["adjust_type"].eq("none")

    # 未复权数据可直接用 amount/volume；复权数据优先保留下载层已调整的 VWAP。
    df["vwap"] = df["raw_vwap"]
    df.loc[unadjusted & calculated_vwap.notna(), "vwap"] = calculated_vwap
    df["vwap"] = df["vwap"].fillna(calculated_vwap)

    vwap_tolerance = pd.concat(
        [
            pd.Series(0.01, index=df.index),
            df["close"].abs() * 0.0001,
        ],
        axis=1,
    ).max(axis=1)
    df["vwap_valid"] = (
        df["vwap"].notna()
        & df["vwap"].gt(0)
        & df["vwap"].ge(df["low"] - vwap_tolerance)
        & df["vwap"].le(df["high"] + vwap_tolerance)
    )

    vwap_difference = (df["raw_vwap"] - calculated_vwap).abs()
    vwap_allowed_error = pd.concat(
        [
            pd.Series(0.01, index=df.index),
            calculated_vwap.abs() * 0.001,
        ],
        axis=1,
    ).max(axis=1)
    vwap_mismatch = (
        unadjusted
        & df["raw_vwap"].notna()
        & calculated_vwap.notna()
        & vwap_difference.gt(vwap_allowed_error)
    )

    # pct_chg 使用百分数；它与 Tushare 的除权昨收口径一致，优先级高于
    # 对未复权 close 直接 pct_change，避免公司行动产生虚假收益率。
    returns_from_pct = df["pct_chg"] / 100.0
    returns_from_pre_close = df["close"].div(
        df["pre_close"].where(df["pre_close"] > 0)
    ) - 1.0
    df["returns"] = returns_from_pct.fillna(returns_from_pre_close)
    df["returns"] = df["returns"].replace([float("inf"), float("-inf")], pd.NA)
    df["returns_valid"] = df["returns"].notna()

    returns_mismatch = (
        df["raw_returns"].notna()
        & df["returns"].notna()
        # 小于万分之一的差异通常来自 pct_chg 显示精度和价格舍入。
        & (df["raw_returns"] - df["returns"])
        .abs()
        .gt(RETURN_MISMATCH_TOLERANCE)
    )

    df["has_missing"] = df[CORE_VALUE_COLUMNS].isna().any(axis=1)
    df["is_tradeable"] = (
        df["ohlc_valid"]
        & df["vwap_valid"]
        & df["volume"].gt(0)
        & df["amount"].gt(0)
    )
    has_trade_status = df["trade_status"].notna()
    df.loc[has_trade_status, "is_tradeable"] &= df.loc[
        has_trade_status, "trade_status"
    ].eq(1)

    metrics = {
        "invalid_ohlc_rows": int((~df["ohlc_valid"]).sum()),
        "invalid_vwap_rows": int((~df["vwap_valid"]).sum()),
        "vwap_mismatch_rows": int(vwap_mismatch.sum()),
        "returns_mismatch_rows": int(returns_mismatch.sum()),
        "missing_rows": int(df["has_missing"].sum()),
        "untradeable_rows": int((~df["is_tradeable"]).sum()),
    }
    return df, metrics


def process_one_file(file_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """读取并处理一只股票，返回标准长表分块和质量记录。"""
    raw = pd.read_parquet(file_path)
    rows_raw = len(raw)
    if raw.empty:
        raise ValueError("原始 Parquet 为空")

    data, unit_action = prepare_base_columns(raw)
    invalid_key_mask = data["code"].isna() | data["date"].isna()
    invalid_key_rows = int(invalid_key_mask.sum())
    data = data.loc[~invalid_key_mask].copy()

    duplicate_mask = data.duplicated(subset=["code", "date"], keep="last")
    duplicates_removed = int(duplicate_mask.sum())
    data = data.loc[~duplicate_mask].copy()
    data = data.sort_values(["code", "date"]).reset_index(drop=True)

    codes = sorted(data["code"].dropna().astype(str).unique().tolist())
    if len(codes) != 1:
        raise ValueError(f"逐股票文件应只包含一个代码，实际为：{codes}")

    data, metrics = build_quality_flags(data)
    data["source_file"] = file_path.name

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("float64")
    for column in [
        "is_tradeable",
        "has_missing",
        "ohlc_valid",
        "vwap_valid",
        "returns_valid",
    ]:
        data[column] = data[column].fillna(False).astype(bool)
    data["adjust_type"] = data["adjust_type"].fillna("unknown").astype(str)
    data["source_file"] = data["source_file"].astype(str)
    data = data[OUTPUT_COLUMNS]

    valid_dates = data["date"].dropna()
    report = {
        "source_file": file_path.name,
        "status": "success",
        "rows_raw": rows_raw,
        "rows_output": len(data),
        "invalid_key_rows": invalid_key_rows,
        "duplicates_removed": duplicates_removed,
        "cross_file_duplicates_removed": 0,
        **metrics,
        "code_count": len(codes),
        "codes": ",".join(codes),
        "start_date": valid_dates.min().strftime("%Y-%m-%d") if not valid_dates.empty else "",
        "end_date": valid_dates.max().strftime("%Y-%m-%d") if not valid_dates.empty else "",
        "adjust_types": ",".join(sorted(data["adjust_type"].unique().tolist())),
        "unit_action": unit_action,
        "missing_by_column": json.dumps(
            {column: int(data[column].isna().sum()) for column in OUTPUT_COLUMNS},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "error_message": "",
    }
    return data, report


def file_sort_key(file_path: Path) -> tuple[str, str]:
    normalized = normalize_stock_code(file_path.stem)
    if pd.isna(normalized):
        return file_path.stem.upper(), file_path.name
    return str(normalized), file_path.name


def failed_report(file_path: Path, error: Exception) -> dict[str, Any]:
    report = {column: "" for column in REPORT_COLUMNS}
    report.update(
        {
            "source_file": file_path.name,
            "status": "failed",
            "rows_raw": 0,
            "rows_output": 0,
            "error_message": str(error),
        }
    )
    return report


def build_total_report(
    reports: list[dict[str, Any]],
    total_codes: set[str],
) -> dict[str, Any]:
    successful = [item for item in reports if item["status"] == "success"]
    total = {column: "" for column in REPORT_COLUMNS}
    count_columns = [
        "rows_raw",
        "rows_output",
        "invalid_key_rows",
        "duplicates_removed",
        "cross_file_duplicates_removed",
        "missing_rows",
        "invalid_ohlc_rows",
        "invalid_vwap_rows",
        "vwap_mismatch_rows",
        "returns_mismatch_rows",
        "untradeable_rows",
    ]
    total.update(
        {
            "source_file": "__TOTAL__",
            "status": "success" if len(successful) == len(reports) else "failed",
            "code_count": len(total_codes),
            "codes": ",".join(sorted(total_codes)),
            "start_date": min(
                (item["start_date"] for item in successful if item["start_date"]),
                default="",
            ),
            "end_date": max(
                (item["end_date"] for item in successful if item["end_date"]),
                default="",
            ),
            "adjust_types": ",".join(
                sorted(
                    {
                        value
                        for item in successful
                        for value in str(item["adjust_types"]).split(",")
                        if value
                    }
                )
            ),
            "error_message": "" if len(successful) == len(reports) else "存在处理失败的原始文件",
        }
    )
    for column in count_columns:
        total[column] = sum(int(item.get(column, 0) or 0) for item in successful)
    return total


def write_quality_report(
    reports: list[dict[str, Any]],
    total_codes: set[str],
    report_file: Path,
) -> pd.DataFrame:
    total = build_total_report(reports, total_codes)
    report_df = pd.DataFrame([*reports, total], columns=REPORT_COLUMNS)
    report_df.to_csv(report_file, index=False, encoding="utf-8-sig")
    return report_df


def preprocess(paths: ProjectPaths, overwrite: bool) -> pd.DataFrame:
    """逐文件处理并流式写入一个 Parquet，避免一次载入全部股票。"""
    if not paths.raw_daily_dir.exists():
        raise FileNotFoundError(f"原始数据目录不存在：{paths.raw_daily_dir}")

    files = sorted(paths.raw_daily_dir.glob("*.parquet"), key=file_sort_key)
    if not files:
        raise FileNotFoundError(f"没有找到原始 Parquet：{paths.raw_daily_dir}")

    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    if paths.output_file.exists() and not overwrite:
        raise FileExistsError(
            f"输出文件已存在：{paths.output_file}\n"
            "确认需要覆盖时，请使用 --overwrite。"
        )

    temp_output = paths.output_file.with_suffix(".tmp.parquet")
    temp_output.unlink(missing_ok=True)

    reports: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, pd.Timestamp]] = set()
    total_codes: set[str] = set()
    writer: pq.ParquetWriter | None = None

    try:
        writer = pq.ParquetWriter(
            temp_output,
            OUTPUT_SCHEMA,
            compression="snappy",
            use_dictionary=["code", "adjust_type", "source_file"],
        )

        for file_path in tqdm(files, desc="预处理日频数据", unit="stock"):
            try:
                data, report = process_one_file(file_path)

                keys = list(zip(data["code"].astype(str), data["date"]))
                cross_duplicate_mask = pd.Series(
                    [key in seen_keys for key in keys],
                    index=data.index,
                )
                cross_duplicate_count = int(cross_duplicate_mask.sum())
                if cross_duplicate_count:
                    data = data.loc[~cross_duplicate_mask].copy()
                    report["cross_file_duplicates_removed"] = cross_duplicate_count
                    report["rows_output"] = len(data)

                seen_keys.update(zip(data["code"].astype(str), data["date"]))
                total_codes.update(data["code"].astype(str).unique().tolist())

                if data.empty:
                    raise ValueError("去除无效键和重复键后没有可写入数据")

                table = pa.Table.from_pandas(
                    data,
                    schema=OUTPUT_SCHEMA,
                    preserve_index=False,
                    safe=False,
                )
                writer.write_table(table)
                reports.append(report)
            except Exception as exc:
                reports.append(failed_report(file_path, exc))

        writer.close()
        writer = None

        report_df = write_quality_report(reports, total_codes, paths.report_file)
        failed_count = int((report_df.iloc[:-1]["status"] == "failed").sum())
        if failed_count:
            temp_output.unlink(missing_ok=True)
            raise RuntimeError(
                f"有 {failed_count} 个原始文件处理失败，未生成不完整的长表。"
                f"请查看：{paths.report_file}"
            )

        temp_output.replace(paths.output_file)
        return report_df
    except Exception:
        if writer is not None:
            writer.close()
        temp_output.unlink(missing_ok=True)
        raise


def print_summary(report_df: pd.DataFrame, paths: ProjectPaths) -> None:
    total = report_df.loc[report_df["source_file"] == "__TOTAL__"].iloc[0]
    print("\n预处理完成")
    print(f"股票数量：{total['code_count']}")
    print(f"原始行数：{total['rows_raw']}")
    print(f"输出行数：{total['rows_output']}")
    print(f"缺失值行数：{total['missing_rows']}")
    print(f"OHLC 异常行数：{total['invalid_ohlc_rows']}")
    print(f"VWAP 异常行数：{total['invalid_vwap_rows']}")
    print(f"原始 returns 口径不一致行数：{total['returns_mismatch_rows']}")
    print(f"不可交易行数：{total['untradeable_rows']}")
    print(f"长表：{paths.output_file}")
    print(f"质量报告：{paths.report_file}")


def main() -> None:
    args = parse_args()
    paths = get_project_paths()
    report_df = preprocess(paths, overwrite=args.overwrite)
    print_summary(report_df, paths)


if __name__ == "__main__":
    main()
