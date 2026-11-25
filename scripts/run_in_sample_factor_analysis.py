"""逐个读取101个因子，执行2018至2022年的样本内单因子分析。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import factor_analysis as fa  # noqa: E402
from alpha_registry import AlphaSpec, select_alpha_specs  # noqa: E402
from factor_store import (  # noqa: E402
    load_factor_source_metadata,
    resolve_factor_path,
)
from sample_periods import IN_SAMPLE_PERIOD, WARMUP_PERIOD  # noqa: E402


OUTPUT_FILENAMES = (
    "factor_summary.csv",
    "yearly_factor_summary.csv",
    "daily_ic.parquet",
    "daily_rank_ic.parquet",
    "adjusted_daily_ic.parquet",
    "adjusted_daily_rank_ic.parquet",
    "factor_coverage.parquet",
    "group_returns.parquet",
    "long_short_returns.parquet",
    "analysis_metadata.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行WorldQuant 101 Alphas的样本内单因子分析"
    )
    parser.add_argument(
        "--bars-file",
        type=Path,
        default=Path("data/processed/daily_bars.parquet"),
        help="处理后日行情长表",
    )
    parser.add_argument(
        "--factors-dir",
        type=Path,
        default=Path("data/processed/alphas"),
        help="逐因子Parquet目录",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/in_sample"),
        help="样本内分析结果目录",
    )
    parser.add_argument(
        "--universe-file",
        type=Path,
        default=Path(
            "data/processed/universe/csi500_daily_membership.parquet"
        ),
        help="逐日历史中证500成员长表",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        default=None,
        help="仅分析指定因子；默认分析注册表中的全部101个因子",
    )
    parser.add_argument("--min-obs", type=int, default=30)
    parser.add_argument("--n-groups", type=int, default=5)
    parser.add_argument("--annualization-factor", type=int, default=252)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖结果目录中的同名分析文件",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_close_panel(bars_file: Path) -> pd.DataFrame:
    """从处理后长表读取收盘价并构造date x code宽表。"""
    if not bars_file.exists():
        raise FileNotFoundError(f"未找到处理后行情文件：{bars_file}")
    bars = pd.read_parquet(bars_file, columns=["date", "code", "close"])
    bars = bars.copy()
    bars["date"] = pd.to_datetime(bars["date"], errors="raise")
    if bars.duplicated(["date", "code"]).any():
        raise ValueError("处理后行情存在重复date-code记录")
    close = bars.pivot(index="date", columns="code", values="close").sort_index()
    close = close.sort_index(axis=1)
    close.index = pd.DatetimeIndex(close.index)
    return close


def load_factor_panel(
    factor_file: Path,
    *,
    expected_index: pd.DatetimeIndex,
    expected_columns: pd.Index,
) -> pd.DataFrame:
    """读取一个因子宽表，并拒绝与行情静默取交集。"""
    if not factor_file.exists():
        raise FileNotFoundError(f"未找到因子文件：{factor_file}")
    factor = pd.read_parquet(factor_file)
    if not isinstance(factor.index, pd.DatetimeIndex):
        factor.index = pd.DatetimeIndex(
            pd.to_datetime(factor.index, errors="raise")
        )
    if not factor.index.equals(expected_index):
        raise ValueError(f"{factor_file.name}的日期索引与行情不一致")
    if not factor.columns.equals(expected_columns):
        raise ValueError(f"{factor_file.name}的股票代码列与行情不一致")
    return factor


def load_universe_panel(
    universe_file: Path,
    *,
    expected_index: pd.DatetimeIndex,
    expected_columns: pd.Index,
) -> pd.DataFrame:
    """读取历史成分股长表并构造与行情严格对齐的布尔宽表。"""
    if not universe_file.exists():
        raise FileNotFoundError(f"未找到历史股票池文件：{universe_file}")
    membership = pd.read_parquet(
        universe_file,
        columns=["date", "code", "is_member"],
    )
    membership = membership.copy()
    membership["date"] = pd.to_datetime(membership["date"], errors="raise")
    membership["code"] = membership["code"].astype("string").str.upper()
    if membership.duplicated(["date", "code"]).any():
        raise ValueError("历史股票池存在重复date-code记录")
    membership["is_member"] = membership["is_member"].astype(bool)
    panel = membership.pivot(
        index="date",
        columns="code",
        values="is_member",
    )
    panel = panel.reindex(
        index=expected_index,
        columns=expected_columns,
        fill_value=False,
    ).fillna(False).astype(bool)
    if panel.sum(axis=1).eq(0).any():
        raise ValueError("历史股票池在部分行情日期没有任何成员")
    return panel


def prepare_in_sample_returns(close: pd.DataFrame) -> pd.DataFrame:
    """仅在样本内价格区间计算未来收益，禁止标签跨入2023年。"""
    in_sample_close = IN_SAMPLE_PERIOD.select(close)
    return fa.calculate_forward_returns(in_sample_close, periods=1)


def _indexed_frame(frame: pd.DataFrame, *, factor_name: str) -> pd.DataFrame:
    result = frame.rename_axis("date").reset_index()
    result.insert(1, "factor_name", factor_name)
    return result


def _empty_failure_summary(spec: AlphaSpec, error: BaseException) -> pd.Series:
    return pd.Series(
        {
            "factor_name": spec.function_name,
            "alpha_id": spec.alpha_id,
            "implementation_batch": spec.implementation_batch,
            "is_approximate": spec.is_approximate,
            "analysis_status": "failed",
            "selection_status": "reject",
            "selection_reason": "原分析状态为failed，无法计算方向调整指标",
            "error_message": f"{type(error).__name__}: {error}",
        }
    )


def analyze_factor_files(
    specs: list[AlphaSpec],
    *,
    close: pd.DataFrame,
    universe_mask: pd.DataFrame,
    factors_dir: Path,
    min_obs: int,
    n_groups: int,
    annualization_factor: int,
) -> dict[str, pd.DataFrame]:
    """逐因子读取、计算和释放宽表，因子失败时继续处理下一项。"""
    forward_returns = prepare_in_sample_returns(close)
    in_sample_dates = forward_returns.index
    in_sample_universe = IN_SAMPLE_PERIOD.select(universe_mask)
    if not in_sample_universe.index.equals(in_sample_dates):
        raise ValueError("样本内历史股票池日期与未来收益日期不一致")
    forward_returns = forward_returns.where(in_sample_universe)

    summaries: list[pd.Series] = []
    yearly_frames: list[pd.DataFrame] = []
    daily_ic: dict[str, pd.Series] = {}
    daily_rank_ic: dict[str, pd.Series] = {}
    adjusted_daily_ic: dict[str, pd.Series] = {}
    adjusted_daily_rank_ic: dict[str, pd.Series] = {}
    coverage_frames: list[pd.DataFrame] = []
    group_frames: list[pd.DataFrame] = []
    long_short_frames: list[pd.DataFrame] = []
    source_metadata = load_factor_source_metadata(factors_dir).set_index(
        "factor_name"
    )

    for spec in tqdm(specs, desc="样本内单因子分析", unit="factor"):
        factor_file = resolve_factor_path(factors_dir, spec.function_name)
        try:
            factor = load_factor_panel(
                factor_file,
                expected_index=close.index,
                expected_columns=close.columns,
            )
            in_sample_factor = IN_SAMPLE_PERIOD.select(factor)
            if not in_sample_factor.index.equals(in_sample_dates):
                raise ValueError("样本内因子日期与未来收益日期不一致")

            analysis = fa.analyze_single_factor(
                in_sample_factor,
                forward_returns,
                factor_name=spec.function_name,
                min_obs=min_obs,
                n_groups=n_groups,
                annualization_factor=annualization_factor,
                universe_mask=in_sample_universe,
            )
            summary = analysis["summary"].copy()
            summary["alpha_id"] = spec.alpha_id
            summary["implementation_batch"] = spec.implementation_batch
            summary["is_approximate"] = spec.is_approximate
            summary["calculation_method"] = source_metadata.at[
                spec.function_name, "calculation_method"
            ]
            summary["is_signal_anchor_exact"] = bool(
                source_metadata.at[
                    spec.function_name, "is_signal_anchor_exact"
                ]
            )
            summary["error_message"] = ""
            summaries.append(summary)

            yearly = analysis["yearly_summary"].reset_index()
            yearly.insert(0, "factor_name", spec.function_name)
            yearly.insert(0, "alpha_id", spec.alpha_id)
            yearly_frames.append(yearly)

            daily_ic[spec.function_name] = analysis["daily_ic"]
            daily_rank_ic[spec.function_name] = analysis["daily_rank_ic"]
            adjusted_daily_ic[spec.function_name] = analysis[
                "adjusted_daily_ic"
            ]
            adjusted_daily_rank_ic[spec.function_name] = analysis[
                "adjusted_daily_rank_ic"
            ]
            coverage_frames.append(
                _indexed_frame(
                    analysis["coverage_details"],
                    factor_name=spec.function_name,
                )
            )
            group_frames.append(
                _indexed_frame(
                    pd.concat(
                        [
                            analysis["group_returns"],
                            analysis["adjusted_group_returns"],
                        ],
                        axis=1,
                    ),
                    factor_name=spec.function_name,
                )
            )
            long_short = pd.concat(
                [
                    analysis["long_short_returns"],
                    analysis["long_short_cumulative_net_value"],
                    analysis["adjusted_long_short_returns"],
                    analysis["adjusted_long_short_cumulative_net_value"],
                ],
                axis=1,
            )
            long_short["selected_direction"] = summary["selected_direction"]
            long_short_frames.append(
                _indexed_frame(long_short, factor_name=spec.function_name)
            )
        except Exception as error:  # 单因子失败不应中断另外100个因子
            summaries.append(_empty_failure_summary(spec, error))

    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary = summary.sort_values("alpha_id").reset_index(drop=True)
    yearly_summary = (
        pd.concat(yearly_frames, ignore_index=True)
        if yearly_frames
        else pd.DataFrame()
    )
    daily_ic_frame = pd.DataFrame(daily_ic, index=in_sample_dates)
    daily_rank_ic_frame = pd.DataFrame(daily_rank_ic, index=in_sample_dates)
    adjusted_daily_ic_frame = pd.DataFrame(
        adjusted_daily_ic,
        index=in_sample_dates,
    )
    adjusted_daily_rank_ic_frame = pd.DataFrame(
        adjusted_daily_rank_ic,
        index=in_sample_dates,
    )
    daily_ic_frame.index.name = "date"
    daily_rank_ic_frame.index.name = "date"
    adjusted_daily_ic_frame.index.name = "date"
    adjusted_daily_rank_ic_frame.index.name = "date"

    return {
        "factor_summary": summary,
        "yearly_factor_summary": yearly_summary,
        "daily_ic": daily_ic_frame,
        "daily_rank_ic": daily_rank_ic_frame,
        "adjusted_daily_ic": adjusted_daily_ic_frame,
        "adjusted_daily_rank_ic": adjusted_daily_rank_ic_frame,
        "factor_coverage": (
            pd.concat(coverage_frames, ignore_index=True)
            if coverage_frames
            else pd.DataFrame()
        ),
        "group_returns": (
            pd.concat(group_frames, ignore_index=True)
            if group_frames
            else pd.DataFrame()
        ),
        "long_short_returns": (
            pd.concat(long_short_frames, ignore_index=True)
            if long_short_frames
            else pd.DataFrame()
        ),
    }


def _assert_only_in_sample(results: dict[str, pd.DataFrame]) -> None:
    """保存前再次确认所有逐日和逐年结果均未进入样本外。"""
    for key in (
        "daily_ic",
        "daily_rank_ic",
        "adjusted_daily_ic",
        "adjusted_daily_rank_ic",
    ):
        frame = results[key]
        if not frame.empty and (
            frame.index.min() < IN_SAMPLE_PERIOD.start
            or frame.index.max() > IN_SAMPLE_PERIOD.end
        ):
            raise AssertionError(f"{key}包含样本内区间以外的日期")
    for key in ("factor_coverage", "group_returns", "long_short_returns"):
        frame = results[key]
        if frame.empty:
            continue
        dates = pd.to_datetime(frame["date"], errors="raise")
        if dates.min() < IN_SAMPLE_PERIOD.start or dates.max() > IN_SAMPLE_PERIOD.end:
            raise AssertionError(f"{key}包含样本内区间以外的日期")
    yearly = results["yearly_factor_summary"]
    if not yearly.empty and not yearly["year"].isin(range(2018, 2023)).all():
        raise AssertionError("年度汇总包含2018至2022以外的年份")


def save_results(
    results: dict[str, pd.DataFrame],
    results_dir: Path,
    *,
    overwrite: bool,
    min_obs: int,
    n_groups: int,
    annualization_factor: int,
    universe_file: Path,
) -> None:
    """保存样本内结果，并阻止意外覆盖其他研究阶段的数据。"""
    existing = [
        results_dir / name
        for name in OUTPUT_FILENAMES
        if (results_dir / name).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"结果文件已存在，请使用--overwrite：{existing[0]}"
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    _assert_only_in_sample(results)

    results["factor_summary"].to_csv(
        results_dir / "factor_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    results["yearly_factor_summary"].to_csv(
        results_dir / "yearly_factor_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    results["daily_ic"].to_parquet(
        results_dir / "daily_ic.parquet",
        index=True,
    )
    results["daily_rank_ic"].to_parquet(
        results_dir / "daily_rank_ic.parquet",
        index=True,
    )
    results["adjusted_daily_ic"].to_parquet(
        results_dir / "adjusted_daily_ic.parquet",
        index=True,
    )
    results["adjusted_daily_rank_ic"].to_parquet(
        results_dir / "adjusted_daily_rank_ic.parquet",
        index=True,
    )
    for key in ("factor_coverage", "group_returns", "long_short_returns"):
        results[key].to_parquet(results_dir / f"{key}.parquet", index=False)

    metadata = pd.DataFrame(
        [
            {
                "analysis_stage": IN_SAMPLE_PERIOD.name,
                "warmup_start": WARMUP_PERIOD.start,
                "warmup_end": WARMUP_PERIOD.end,
                "analysis_start": IN_SAMPLE_PERIOD.start,
                "analysis_end": IN_SAMPLE_PERIOD.end,
                "forward_return_definition": "close[t+1] / close[t] - 1",
                "cross_boundary_forward_return_allowed": False,
                "universe_definition": "CSI300 historical daily membership",
                "universe_file": str(universe_file),
                "universe_count_definition": "daily historical members",
                "min_obs": min_obs,
                "n_groups": n_groups,
                "annualization_factor": annualization_factor,
                "risk_free_rate": 0.0,
                "direction_rule": "sign(in_sample_mean_daily_rank_ic)",
                "min_valid_rank_ic_day_ratio": (
                    fa.MIN_VALID_RANK_IC_DAY_RATIO
                ),
                "min_adjusted_rank_ic": fa.MIN_ADJUSTED_RANK_IC,
                "min_adjusted_rank_icir": fa.MIN_ADJUSTED_RANK_ICIR,
                "min_adjusted_rank_ic_win_rate": (
                    fa.MIN_ADJUSTED_RANK_IC_WIN_RATE
                ),
                "min_positive_rank_ic_years": (
                    fa.MIN_POSITIVE_RANK_IC_YEARS
                ),
                "max_rank_ic_top_5pct_abs_share": (
                    fa.MAX_RANK_IC_TOP_5PCT_ABS_SHARE
                ),
            }
        ]
    )
    metadata.to_csv(
        results_dir / "analysis_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    args = parse_args()
    if args.min_obs < 2:
        raise ValueError("min_obs必须大于等于2")
    if args.n_groups < 2:
        raise ValueError("n_groups必须大于等于2")
    if args.annualization_factor < 1:
        raise ValueError("annualization_factor必须大于等于1")

    bars_file = resolve_project_path(args.bars_file)
    factors_dir = resolve_project_path(args.factors_dir)
    results_dir = resolve_project_path(args.results_dir)
    universe_file = resolve_project_path(args.universe_file)
    specs = select_alpha_specs(args.alphas, implemented_only=True)
    close = load_close_panel(bars_file)
    universe_mask = load_universe_panel(
        universe_file,
        expected_index=close.index,
        expected_columns=close.columns,
    )
    results = analyze_factor_files(
        specs,
        close=close,
        universe_mask=universe_mask,
        factors_dir=factors_dir,
        min_obs=args.min_obs,
        n_groups=args.n_groups,
        annualization_factor=args.annualization_factor,
    )
    save_results(
        results,
        results_dir,
        overwrite=args.overwrite,
        min_obs=args.min_obs,
        n_groups=args.n_groups,
        annualization_factor=args.annualization_factor,
        universe_file=universe_file,
    )

    analysis_statuses = results["factor_summary"][
        "analysis_status"
    ].value_counts()
    selection_statuses = results["factor_summary"][
        "selection_status"
    ].value_counts()
    print("样本内单因子分析完成。")
    print(
        f"统计区间：{IN_SAMPLE_PERIOD.start.date()} 至 "
        f"{IN_SAMPLE_PERIOD.end.date()}"
    )
    print("未来收益不会跨越到2023年。")
    print("分析状态：")
    print(analysis_statuses.to_string())
    print("筛选状态：")
    print(selection_statuses.to_string())
    print(f"结果目录：{results_dir}")


if __name__ == "__main__":
    main()
