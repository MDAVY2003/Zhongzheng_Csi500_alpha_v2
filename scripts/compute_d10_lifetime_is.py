"""按 D10 曲线计算因子寿命（仅 IS 2018-2022）。

规则（已与用户确认）：
- 方向由 D10 曲线的峰值决定；
- normal（峰值在 D1）：寿命 = D10 半衰期；
- delayed（峰值在 D1 之后）：取峰值附近 D10 >= 0.5*peak 的连续半高宽窗口，
  记录 start/end/width；delayed 不再细分，统一 bucket=delayed；
- 只用 IS 数据；h 日前瞻收益在 2022-12-31 截断，不使用 2023 年价格；
- 不做任何筛选/门槛，只输出全部因子的寿命和分组。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_market(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = ROOT / "data/processed/research_market_v1"
    qo = pd.read_parquet(data / "qfq_open.parquet").where(lambda x: x > 0)
    mem = pd.read_parquet(data / "membership.parquet").astype(bool)
    if not qo.index.equals(mem.index) or not qo.columns.equals(mem.columns):
        raise ValueError("qfq_open 与 membership 轴不一致")
    start, end = cfg["period"]["start"], cfg["period"]["end"]
    qo = qo.loc[start:end]
    mem = mem.loc[start:end]
    return qo, mem


def horizon_stats(
    sig: pd.DataFrame,
    fwd: pd.DataFrame,
    mem: pd.DataFrame,
    h: int,
) -> tuple[float, float, int, int]:
    valid = sig.notna() & fwd.notna() & mem
    n_valid = int(valid.to_numpy().sum())
    n_member = int(mem.to_numpy().sum())
    pct = sig.where(valid).rank(axis=1, method='first', pct=True)
    dec = np.ceil(pct * 10).clip(1, 10)
    ew = fwd.where(valid).mean(axis=1)
    ex = fwd.where(valid).sub(ew, axis=0)
    d1 = float(ex.where(dec.eq(1)).mean(axis=1).mean() * 252 / h)
    d10 = float(ex.where(dec.eq(10)).mean(axis=1).mean() * 252 / h)
    return d1, d10, n_valid, n_member


def compute_factor(
    name: str,
    panel: pd.DataFrame,
    qo: pd.DataFrame,
    mem: pd.DataFrame,
    fwd: dict[int, pd.DataFrame],
    horizons: list[int],
    cfg: dict,
) -> tuple[dict, list[dict]]:
    sig = panel.shift(1).reindex(index=qo.index, columns=qo.columns).where(mem)
    d1 = {}
    d10 = {}
    valid_counts = {}
    member_counts = {}
    for h in horizons:
        a, b, nv, nm = horizon_stats(sig, fwd[h], mem, h)
        d1[h] = a
        d10[h] = b
        valid_counts[h] = nv
        member_counts[h] = nm

    pos = np.array([d10[h] for h in horizons], dtype=float)
    neg = np.array([d1[h] for h in horizons], dtype=float)
    pos_peak = float(np.nanmax(pos)) if np.isfinite(pos).any() else np.nan
    neg_peak = float(np.nanmax(neg)) if np.isfinite(neg).any() else np.nan

    if not np.isfinite(pos_peak) and not np.isfinite(neg_peak):
        return {
            "factor": name,
            "direction": np.nan,
            "peak_h": np.nan,
            "peak_d10": np.nan,
            "lifetime": np.nan,
            "lifetime_width": np.nan,
            "window_start": np.nan,
            "window_end": np.nan,
            "bucket": "unclassified",
            "status": "all_nan",
        }, []

    direction = 1.0 if (not np.isfinite(neg_peak) or pos_peak >= neg_peak) else -1.0
    curve = pos if direction > 0 else neg
    peak_idx = int(np.nanargmax(curve))
    peak_h = peak_idx + 1
    peak = float(curve[peak_idx])

    row = {
        "factor": name,
        "direction": direction,
        "peak_h": peak_h,
        "peak_d10": peak,
        "coverage_h1": (
            valid_counts[1] / member_counts[1] if member_counts[1] else np.nan
        ),
        "n_valid_h1": valid_counts[1],
        "n_member_h1": member_counts[1],
    }

    if not np.isfinite(peak) or peak <= 0:
        row.update(
            lifetime=np.nan,
            lifetime_width=np.nan,
            window_start=np.nan,
            window_end=np.nan,
            bucket="unclassified",
            status="invalid_peak",
            lifetime_definition="",
            right_censored=False,
            boundary_censored=False,
        )
    elif peak_h == 1:
        threshold = cfg["normal_half_life_threshold"] * peak
        lifetime = None
        boundary_censored = False
        right_censored = False
        for h in horizons[1:]:
            v = curve[h - 1]
            if not np.isfinite(v):
                lifetime = h - 1
                boundary_censored = True
                break
            if v < threshold:
                lifetime = h - 1
                break
        if lifetime is None:
            lifetime = cfg["lifetime_cap"]
            right_censored = True
        if lifetime <= 2:
            bucket = "short"
        elif lifetime <= 5:
            bucket = "middle"
        else:
            bucket = "long"
        row.update(
            lifetime=lifetime,
            lifetime_width=lifetime,
            window_start=1,
            window_end=lifetime,
            bucket=bucket,
            status="success",
            lifetime_definition="normal_half_life",
            right_censored=right_censored,
            boundary_censored=boundary_censored,
        )
    else:
        threshold = cfg["delayed_half_height_threshold"] * peak
        above = curve >= threshold
        start = peak_h
        while start > 1 and bool(above[start - 2]):
            start -= 1
        end = peak_h
        while end < cfg["lifetime_cap"] and bool(above[end]):
            end += 1
        width = end - start + 1
        right_censored = bool(end == cfg["lifetime_cap"] and above[end - 1])
        row.update(
            lifetime=end,
            lifetime_width=width,
            window_start=start,
            window_end=end,
            bucket="delayed",
            status="success",
            lifetime_definition="delayed_window_end",
            right_censored=right_censored,
            boundary_censored=False,
        )

    for h in horizons:
        row[f"d10_{h}"] = d10[h] if direction > 0 else d1[h]
        row[f"d1_raw_{h}"] = d1[h]

    curve_rows = []
    for h in horizons:
        curve_rows.append(
            {
                "factor": name,
                "direction": direction,
                "horizon": h,
                "d10": row[f"d10_{h}"],
                "d1_raw": d1[h],
                "valid_stock_days": valid_counts[h],
                "member_stock_days": member_counts[h],
            }
        )
    return row, curve_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/lifetime_is.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factors", type=str, default=None, help="逗号分隔的因子子集，用于测试")
    parser.add_argument("--max-factors", type=int, default=None)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"输出目录已存在：{out}")
    out.mkdir(parents=True)

    qo, mem = load_market(cfg)
    horizons = list(cfg["horizons"])
    fwd = {h: qo.shift(-h) / qo - 1.0 for h in horizons}

    factor_dir = ROOT / "data/processed/alphas_point_in_time"
    names = sorted(p.stem for p in factor_dir.glob("alpha*.parquet"))
    if args.factors:
        names = [x.strip() for x in args.factors.split(",") if x.strip()]
    if args.max_factors:
        names = names[: args.max_factors]
    rows = []
    curves = []
    failures = []
    for i, name in enumerate(names, 1):
        try:
            panel = pd.read_parquet(factor_dir / f"{name}.parquet")
            if panel.empty or panel.notna().sum().sum() == 0:
                failures.append({"factor": name, "reason": "empty_or_all_nan"})
                print(f"[{i}/{len(names)}] {name} 跳过：全空", flush=True)
                continue
            row, curve_rows = compute_factor(
                name, panel, qo, mem, fwd, horizons, cfg
            )
            rows.append(row)
            curves.extend(curve_rows)
            print(
                f"[{i}/{len(names)}] {name} dir={row['direction']:+.0f} "
                f"peak=D{row['peak_h']} {row['peak_d10']*100:+.1f}% "
                f"lifetime={row['lifetime']} bucket={row['bucket']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"factor": name, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[{i}/{len(names)}] {name} 失败：{exc}", flush=True)

    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values(["bucket", "lifetime", "factor"]).reset_index(drop=True)
    result.to_csv(out / "d10_lifetime.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "d10_curves.csv", index=False)
    pd.DataFrame(failures).to_csv(out / "failures.csv", index=False)

    counts = (
        result["bucket"].value_counts().to_dict() if len(result) else {}
    )
    summary = {
        "config": str(args.config),
        "period": cfg["period"],
        "n_factors": len(names),
        "n_computed": int(len(result)),
        "n_failures": len(failures),
        "bucket_counts": counts,
        "status_counts": result["status"].value_counts().to_dict() if len(result) else {},
        "delayed_count": int((result["bucket"] == "delayed").sum()) if len(result) else 0,
        "note": "仅 IS 2018-2022；前瞻收益在 2022-12-31 截断；未做筛选。",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    print("\n完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
