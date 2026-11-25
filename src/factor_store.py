"""统一解析因子文件，并优先使用已经生成的严格信号日锚定版本。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


STRICT_SIGNAL_ANCHOR_FACTORS = frozenset({"alpha004", "alpha067", "alpha088"})
STRICT_DIRECTORY_NAME = "alphas_point_in_time_exact_signal_anchor"


def _parse_bool(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    invalid = ~normalized.isin(["true", "false"])
    if invalid.any():
        raise ValueError(f"{name}包含无法解析的布尔值")
    return normalized.eq("true")


def normalize_factor_name(factor_name: str) -> str:
    """把 alpha_004、alpha4 等写法统一为 alpha004。"""
    normalized = factor_name.lower().replace("_", "")
    if not normalized.startswith("alpha"):
        raise ValueError(f"无效因子名：{factor_name}")
    try:
        alpha_id = int(normalized.removeprefix("alpha"))
    except ValueError as error:
        raise ValueError(f"无效因子名：{factor_name}") from error
    if not 1 <= alpha_id <= 101:
        raise ValueError(f"因子编号必须介于1至101：{factor_name}")
    return f"alpha{alpha_id:03d}"


def strict_factor_directory(base_factors_dir: Path) -> Path:
    """严格版本与旧目录并列保存，禁止覆盖旧因子。"""
    return base_factors_dir.parent / STRICT_DIRECTORY_NAME


def resolve_factor_path(
    base_factors_dir: Path,
    factor_name: str,
    *,
    require_exists: bool = True,
) -> Path:
    """对指定的三个因子强制读取严格版本，其余因子读取旧目录。"""
    name = normalize_factor_name(factor_name)
    if name in STRICT_SIGNAL_ANCHOR_FACTORS:
        path = strict_factor_directory(base_factors_dir) / f"{name}.parquet"
        source = "严格信号日锚定"
    else:
        path = base_factors_dir / f"{name}.parquet"
        source = "基础点时"
    if require_exists and not path.exists():
        raise FileNotFoundError(f"缺少{source}因子文件：{path}")
    return path


def load_factor_source_metadata(base_factors_dir: Path) -> pd.DataFrame:
    """合并基础与严格运行清单，严格版本覆盖同名旧记录。"""
    base_summary = base_factors_dir / "point_in_time_run_summary.csv"
    if not base_summary.exists():
        raise FileNotFoundError(f"缺少基础因子运行清单：{base_summary}")
    base = pd.read_csv(base_summary, encoding="utf-8-sig")
    required = {"factor_name", "calculation_method", "is_signal_anchor_exact"}
    missing = required.difference(base.columns)
    if missing:
        raise ValueError(f"基础因子运行清单缺少字段：{sorted(missing)}")
    metadata = base[list(required)].copy()

    strict_summary = (
        strict_factor_directory(base_factors_dir)
        / "exact_signal_anchor_run_summary.csv"
    )
    if strict_summary.exists():
        strict = pd.read_csv(strict_summary, encoding="utf-8-sig")
        strict_missing = required.difference(strict.columns)
        if strict_missing:
            raise ValueError(f"严格因子运行清单缺少字段：{sorted(strict_missing)}")
        metadata = metadata.loc[
            ~metadata["factor_name"].isin(strict["factor_name"])
        ]
        metadata = pd.concat(
            [metadata, strict[list(required)]],
            ignore_index=True,
        )
    metadata["is_signal_anchor_exact"] = _parse_bool(
        metadata["is_signal_anchor_exact"],
        "is_signal_anchor_exact",
    )
    if metadata["factor_name"].duplicated().any():
        raise ValueError("因子来源元数据存在重复factor_name")
    return metadata.sort_values("factor_name").reset_index(drop=True)
