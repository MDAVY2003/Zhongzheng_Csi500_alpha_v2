"""根据 Alpha 注册表批量计算已实现因子并保存宽表。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import operators as op
from alpha_registry import (
    AlphaSpec,
    save_registry_manifest,
    select_alpha_specs,
)
from factor_data import FactorData, load_cap_panel, load_sw_industry_panels


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BARS_FILE = PROJECT_ROOT / "data" / "processed" / "daily_bars.parquet"
DEFAULT_CAP_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "factor_dependencies"
    / "daily_market_cap.parquet"
)
DEFAULT_INDUSTRY_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "factor_dependencies"
    / "sw_industry_membership.parquet"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "alphas"
DEFAULT_UNIVERSE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "universe"
    / "csi500_daily_membership.parquet"
)
INDUSTRY_FIELDS = {"sector", "industry", "subindustry"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按注册表批量计算WorldQuant 101 Alphas")
    parser.add_argument(
        "--alphas",
        nargs="+",
        default=None,
        help="只运行指定因子，例如 --alphas 12 alpha033 alpha_101",
    )
    parser.add_argument("--bars-file", type=Path, default=DEFAULT_BARS_FILE)
    parser.add_argument("--cap-file", type=Path, default=DEFAULT_CAP_FILE)
    parser.add_argument(
        "--industry-file",
        type=Path,
        default=DEFAULT_INDUSTRY_FILE,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--universe-file",
        type=Path,
        default=DEFAULT_UNIVERSE_FILE,
        help="逐日历史股票池；横截面算子和最终输出只使用当日成员",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="重新计算并覆盖已有因子文件",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="只打印注册表摘要，不读取行情或计算因子",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="只重新检查已保存因子文件，不重新计算",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _required_data_fields(specs: list[AlphaSpec]) -> set[str]:
    required: set[str] = set()
    for spec in specs:
        required.update(spec.required_fields)
        required.update(spec.industry_levels)
    return required


def load_factor_data(
    specs: list[AlphaSpec],
    *,
    bars_file: Path,
    cap_file: Path,
    industry_file: Path,
) -> FactorData:
    """只读取选中已实现因子需要的字段。"""
    if not specs:
        raise ValueError("至少需要一个已实现因子才能加载FactorData")
    if not bars_file.exists():
        raise FileNotFoundError(f"处理后行情不存在：{bars_file}")

    required = _required_data_fields(specs)
    bar_fields = sorted(required.difference({"cap"}, INDUSTRY_FIELDS))
    columns = ["date", "code", *bar_fields]
    try:
        bars = pd.read_parquet(bars_file, columns=columns)
    except Exception as error:
        raise ValueError(
            f"无法从处理后行情读取所需字段{bar_fields}: {error}"
        ) from error

    bars = bars.copy()
    bars["date"] = pd.to_datetime(bars["date"], errors="raise")
    bars["code"] = bars["code"].astype("string").str.upper()
    if bars.duplicated(["date", "code"]).any():
        raise ValueError("处理后行情存在重复date-code")
    bars = bars.sort_values(["date", "code"])

    panels = {
        field: bars.pivot(index="date", columns="code", values=field).sort_index()
        for field in bar_fields
    }
    reference = next(iter(panels.values()))
    dates = reference.index
    codes = reference.columns.astype(str).tolist()

    if "cap" in required:
        if not cap_file.exists():
            raise FileNotFoundError(f"市值数据不存在：{cap_file}")
        panels["cap"] = load_cap_panel(cap_file, dates=dates, codes=codes)

    needed_industry_levels = required.intersection(INDUSTRY_FIELDS)
    if needed_industry_levels:
        if not industry_file.exists():
            raise FileNotFoundError(f"申万行业数据不存在：{industry_file}")
        industry_panels = load_sw_industry_panels(
            industry_file,
            dates=dates,
            codes=codes,
        )
        for level in needed_industry_levels:
            panels[level] = industry_panels[level]

    return FactorData.from_mapping(panels)


def load_universe_mask(
    file_path: Path,
    *,
    dates: pd.DatetimeIndex,
    codes: pd.Index,
) -> pd.DataFrame:
    """读取date-code历史成员长表并构造成与FactorData对齐的布尔宽表。"""
    if not file_path.exists():
        raise FileNotFoundError(f"历史股票池不存在：{file_path}")
    columns = pd.read_parquet(file_path).columns
    requested = ["date", "code"]
    if "is_member" in columns:
        requested.append("is_member")
    membership = pd.read_parquet(file_path, columns=requested)
    membership["date"] = pd.to_datetime(membership["date"], errors="raise")
    membership["code"] = membership["code"].astype("string").str.upper()
    if membership.duplicated(["date", "code"]).any():
        raise ValueError("历史股票池存在重复date-code")
    if "is_member" not in membership:
        membership["is_member"] = True
    membership["is_member"] = membership["is_member"].fillna(False).astype(bool)
    mask = membership.pivot(index="date", columns="code", values="is_member")
    return mask.reindex(index=dates, columns=codes, fill_value=False).fillna(False).astype(bool)


def validate_alpha_output(
    result: pd.DataFrame,
    data: FactorData,
    spec: AlphaSpec,
) -> None:
    """强制因子输出保持 date x code 宽表协议。"""
    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"{spec.function_name}输出必须是pandas DataFrame")
    if not result.index.equals(data.index):
        raise ValueError(f"{spec.function_name}输出日期索引与data不一致")
    if not result.columns.equals(data.columns):
        raise ValueError(f"{spec.function_name}输出股票代码列与data不一致")
    numeric = result.select_dtypes(include=[np.number])
    if numeric.shape[1] != result.shape[1]:
        raise TypeError(f"{spec.function_name}输出必须全部是数值列")
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError(f"{spec.function_name}输出包含正负无穷")


def run_registered_alphas(
    specs: list[AlphaSpec],
    *,
    data: FactorData | None,
    output_dir: Path,
    overwrite: bool = False,
    universe_mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """逐个运行因子，单个因子失败不中断其他因子。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    if universe_mask is not None and data is not None:
        if not universe_mask.index.equals(data.index):
            raise ValueError("历史股票池日期与FactorData不一致")
        if not universe_mask.columns.equals(data.columns):
            raise ValueError("历史股票池股票列与FactorData不一致")
    summary_rows: list[dict[str, object]] = []

    for spec in tqdm(specs, desc="计算Alpha", unit="factor"):
        output_file = output_dir / f"{spec.function_name}.parquet"
        common = {
            "alpha_id": spec.alpha_id,
            "function_name": spec.function_name,
            "implementation_status": spec.status,
            "is_approximate": spec.is_approximate,
            "max_lookback": spec.max_lookback,
            "file_path": str(output_file),
        }

        if spec.status != "implemented" or spec.function is None:
            reason = "因子函数尚未实现"
            spec.record_skipped(reason)
            summary_rows.append(
                {**common, "run_status": "not_implemented", "rows": 0,
                 "columns": 0, "valid_values": 0, "error_message": reason}
            )
            continue

        if data is None:
            reason = "未加载FactorData"
            spec.record_skipped(reason)
            summary_rows.append(
                {**common, "run_status": "skipped", "rows": 0,
                 "columns": 0, "valid_values": 0, "error_message": reason}
            )
            continue

        missing = set(spec.required_fields).union(spec.industry_levels).difference(
            data.available_fields
        )
        if missing:
            reason = f"FactorData缺少字段：{sorted(missing)}"
            spec.record_skipped(reason)
            summary_rows.append(
                {**common, "run_status": "skipped", "rows": 0,
                 "columns": 0, "valid_values": 0, "error_message": reason}
            )
            continue

        if output_file.exists() and not overwrite:
            reason = "文件已存在，未使用--overwrite"
            spec.record_skipped(reason)
            summary_rows.append(
                {**common, "run_status": "skipped", "rows": 0,
                 "columns": 0, "valid_values": 0, "error_message": reason}
            )
            continue

        try:
            with op.use_universe_mask(universe_mask):
                result = spec.function(data)
            if universe_mask is not None:
                result = result.where(universe_mask)
            validate_alpha_output(result, data, spec)
            result = result.copy()
            result.index.name = "date"
            result.to_parquet(output_file, index=True)
            spec.record_success()
            summary_rows.append(
                {
                    **common,
                    "run_status": "success",
                    "rows": result.shape[0],
                    "columns": result.shape[1],
                    "valid_values": int(result.notna().to_numpy().sum()),
                    "error_message": "",
                }
            )
        except Exception as error:  # 单因子失败必须继续运行其他因子。
            spec.record_failure(error)
            summary_rows.append(
                {
                    **common,
                    "run_status": "failed",
                    "rows": 0,
                    "columns": 0,
                    "valid_values": 0,
                    "error_message": spec.last_error,
                }
            )

    return pd.DataFrame(summary_rows).sort_values("alpha_id").reset_index(drop=True)


def print_registry(specs: list[AlphaSpec]) -> None:
    table = pd.DataFrame(
        {
            "alpha_id": [spec.alpha_id for spec in specs],
            "function_name": [spec.function_name for spec in specs],
            "status": [spec.status for spec in specs],
            "batch": [spec.implementation_batch for spec in specs],
            "max_lookback": [spec.max_lookback for spec in specs],
            "required_fields": [",".join(spec.required_fields) for spec in specs],
            "industry_levels": [",".join(spec.industry_levels) for spec in specs],
        }
    )
    print(table.to_string(index=False))


def audit_factor_outputs(
    specs: list[AlphaSpec],
    output_dir: Path,
    universe_mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """重复读取因子宽表并按历史股票池统计覆盖率和越界值。"""
    records: list[dict[str, object]] = []
    for spec in tqdm(specs, desc="审计Alpha", unit="factor"):
        file_path = output_dir / f"{spec.function_name}.parquet"
        if not file_path.exists():
            records.append(
                {
                    "alpha_id": spec.alpha_id,
                    "function_name": spec.function_name,
                    "rows": 0,
                    "columns": 0,
                    "valid_values": 0,
                    "total_values": 0,
                    "coverage": np.nan,
                    "member_values": 0,
                    "member_coverage": np.nan,
                    "outside_member_valid_values": 0,
                    "valid_dates": 0,
                    "first_valid_date": "",
                    "last_valid_date": "",
                    "max_valid_stocks_per_day": 0,
                    "inf_values": 0,
                    "quality_warning": "missing_file",
                }
            )
            continue

        factor = pd.read_parquet(file_path)
        factor.index = pd.to_datetime(factor.index, errors="raise")
        if universe_mask is None:
            member_mask = pd.DataFrame(
                True,
                index=factor.index,
                columns=factor.columns,
            )
        else:
            member_mask = universe_mask.reindex(
                index=factor.index,
                columns=factor.columns,
                fill_value=False,
            ).fillna(False).astype(bool)
        valid = factor.notna()
        valid_in_universe = valid & member_mask
        valid_by_date = valid_in_universe.sum(axis=1)
        valid_dates = valid_by_date[valid_by_date > 0].index
        values = factor.to_numpy(dtype=float)
        valid_values = int(valid_in_universe.to_numpy().sum())
        total_values = int(factor.size)
        member_values = int(member_mask.to_numpy().sum())
        member_coverage = (
            valid_values / member_values if member_values else np.nan
        )
        outside_member_valid_values = int(
            (valid & ~member_mask).to_numpy().sum()
        )
        inf_values = int(np.isinf(values).sum())

        warning = ""
        if valid_values == 0:
            warning = "all_nan"
        elif inf_values:
            warning = "contains_infinity"
        elif outside_member_valid_values:
            warning = "values_outside_universe"
        elif member_coverage < 0.01:
            warning = "very_low_coverage"
        elif member_coverage < 0.10:
            warning = "low_coverage"

        records.append(
            {
                "alpha_id": spec.alpha_id,
                "function_name": spec.function_name,
                "rows": factor.shape[0],
                "columns": factor.shape[1],
                "valid_values": valid_values,
                "total_values": total_values,
                "coverage": member_coverage,
                "member_values": member_values,
                "member_coverage": member_coverage,
                "outside_member_valid_values": outside_member_valid_values,
                "valid_dates": len(valid_dates),
                "first_valid_date": (
                    valid_dates.min().strftime("%Y-%m-%d")
                    if len(valid_dates)
                    else ""
                ),
                "last_valid_date": (
                    valid_dates.max().strftime("%Y-%m-%d")
                    if len(valid_dates)
                    else ""
                ),
                "max_valid_stocks_per_day": int(valid_by_date.max()),
                "inf_values": inf_values,
                "quality_warning": warning,
            }
        )
    return pd.DataFrame(records).sort_values("alpha_id").reset_index(drop=True)


def build_calculation_summary(
    specs: list[AlphaSpec],
    quality: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """结合注册表与文件审计，生成当前全局计算状态。"""
    quality_by_id = quality.set_index("alpha_id")
    records: list[dict[str, object]] = []
    for spec in specs:
        row = quality_by_id.loc[spec.alpha_id]
        warning = str(row["quality_warning"])
        if warning == "missing_file":
            status = "missing"
        elif warning in {"all_nan", "contains_infinity"}:
            status = "invalid"
        else:
            status = "success"
        records.append(
            {
                "alpha_id": spec.alpha_id,
                "function_name": spec.function_name,
                "implementation_batch": spec.implementation_batch,
                "is_approximate": spec.is_approximate,
                "calculation_status": status,
                "rows": int(row["rows"]),
                "columns": int(row["columns"]),
                "valid_values": int(row["valid_values"]),
                "coverage": row["coverage"],
                "member_coverage": row["member_coverage"],
                "outside_member_valid_values": int(
                    row["outside_member_valid_values"]
                ),
                "quality_warning": warning,
                "file_path": str(output_dir / f"{spec.function_name}.parquet"),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    specs = select_alpha_specs(args.alphas)
    output_dir = resolve_project_path(args.output_dir)
    if args.list:
        print_registry(specs)
        return

    implemented_specs = [spec for spec in specs if spec.status == "implemented"]
    if args.audit_only:
        universe_mask = None
        if implemented_specs:
            first_file = output_dir / f"{implemented_specs[0].function_name}.parquet"
            if first_file.exists():
                reference = pd.read_parquet(first_file)
                universe_mask = load_universe_mask(
                    resolve_project_path(args.universe_file),
                    dates=pd.DatetimeIndex(reference.index),
                    codes=reference.columns,
                )
        quality = audit_factor_outputs(
            implemented_specs,
            output_dir,
            universe_mask=universe_mask,
        )
        quality_file = output_dir / "factor_output_quality.csv"
        quality.to_csv(quality_file, index=False, encoding="utf-8-sig")
        calculation_summary = build_calculation_summary(
            implemented_specs, quality, output_dir
        )
        calculation_summary_file = output_dir / "calculation_summary.csv"
        calculation_summary.to_csv(
            calculation_summary_file, index=False, encoding="utf-8-sig"
        )
        print(f"数据质量报告：{quality_file}")
        print(f"全局计算状态：{calculation_summary_file}")
        warnings = quality[quality["quality_warning"] != ""]
        if not warnings.empty:
            print(warnings.to_string(index=False))
        return

    data = None
    universe_mask = None
    if implemented_specs:
        data = load_factor_data(
            implemented_specs,
            bars_file=resolve_project_path(args.bars_file),
            cap_file=resolve_project_path(args.cap_file),
            industry_file=resolve_project_path(args.industry_file),
        )
        universe_mask = load_universe_mask(
            resolve_project_path(args.universe_file),
            dates=data.index,
            codes=data.columns,
        )

    summary = run_registered_alphas(
        specs,
        data=data,
        output_dir=output_dir,
        overwrite=args.overwrite,
        universe_mask=universe_mask,
    )
    summary_file = output_dir / "run_summary.csv"
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")
    save_registry_manifest(output_dir / "alpha_registry_runtime.csv")
    all_implemented_specs = select_alpha_specs(implemented_only=True)
    quality = audit_factor_outputs(
        all_implemented_specs,
        output_dir,
        universe_mask=universe_mask,
    )
    quality_file = output_dir / "factor_output_quality.csv"
    quality.to_csv(quality_file, index=False, encoding="utf-8-sig")
    calculation_summary = build_calculation_summary(
        all_implemented_specs, quality, output_dir
    )
    calculation_summary_file = output_dir / "calculation_summary.csv"
    calculation_summary.to_csv(
        calculation_summary_file, index=False, encoding="utf-8-sig"
    )

    print("\n运行汇总：")
    print(summary["run_status"].value_counts(dropna=False).to_string())
    print(f"因子目录：{output_dir}")
    print(f"汇总文件：{summary_file}")
    print(f"数据质量报告：{quality_file}")
    print(f"全局计算状态：{calculation_summary_file}")


if __name__ == "__main__":
    main()
