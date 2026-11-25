"""Recompute D10 curves and lifetimes with the v3 ex-ante rules.

Only IS 2018-2022.  No 2023+ prices are used in labels: forward returns are
computed on the IS slice, so the last h rows are NaN.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from d10_core import choose_direction, compute_horizon  # noqa: E402


def load_market() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = ROOT / "data/processed/research_market_v1"
    qo = pd.read_parquet(data / "qfq_open.parquet").where(lambda x: x > 0)
    mem = pd.read_parquet(data / "membership.parquet").astype(bool)
    qo = qo.loc["2018-01-01":"2022-12-31"]
    mem = mem.loc["2018-01-01":"2022-12-31"]
    if not qo.index.equals(mem.index) or not qo.columns.equals(mem.columns):
        raise ValueError("qfq_open 与 membership 轴不一致")
    return qo, mem


def lifetime_from_curve(curve: np.ndarray, cap: int = 10) -> dict:
    if not np.isfinite(curve).any():
        return dict(peak_h=np.nan, peak_d10=np.nan, lifetime=np.nan,
                    lifetime_width=np.nan, window_start=np.nan, window_end=np.nan,
                    bucket="unclassified", status="all_nan", right_censored=False,
                    boundary_censored=False)
    idx = int(np.nanargmax(curve))
    peak_h = idx + 1
    peak = float(curve[idx])
    if not np.isfinite(peak) or peak <= 0:
        return dict(peak_h=peak_h, peak_d10=peak, lifetime=np.nan,
                    lifetime_width=np.nan, window_start=np.nan, window_end=np.nan,
                    bucket="unclassified", status="invalid_peak", right_censored=False,
                    boundary_censored=False)
    if peak_h == 1:
        threshold = 0.5 * peak
        lifetime = None
        boundary = False
        right = False
        for h in range(2, cap + 1):
            v = curve[h - 1]
            if not np.isfinite(v):
                lifetime = h - 1
                boundary = True
                break
            if v < threshold:
                lifetime = h - 1
                break
        if lifetime is None:
            lifetime = cap
            right = True
        bucket = "short" if lifetime <= 2 else "middle" if lifetime <= 5 else "long"
        return dict(peak_h=1, peak_d10=peak, lifetime=int(lifetime),
                    lifetime_width=int(lifetime), window_start=1,
                    window_end=int(lifetime), bucket=bucket, status="success",
                    right_censored=right, boundary_censored=boundary)
    threshold = 0.5 * peak
    above = curve >= threshold
    start = peak_h
    while start > 1 and bool(above[start - 2]):
        start -= 1
    end = peak_h
    while end < cap and bool(above[end]):
        end += 1
    width = end - start + 1
    right = bool(end == cap and above[end - 1])
    return dict(peak_h=peak_h, peak_d10=peak, lifetime=int(end),
                lifetime_width=int(width), window_start=int(start),
                window_end=int(end), bucket="delayed", status="success",
                right_censored=right, boundary_censored=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factors", type=str, default=None)
    parser.add_argument("--max-factors", type=int, default=None)
    parser.add_argument("--factor-dir", type=Path, default=ROOT / "data/processed/alphas_point_in_time")
    parser.add_argument("--provenance", choices=["base", "strict"], default="base")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"输出目录已存在：{out}")
    out.mkdir(parents=True)

    qo, mem = load_market()
    horizons = list(range(1, 11))
    fwd = {h: qo.shift(-h) / qo - 1.0 for h in horizons}

    factor_dir = args.factor_dir.resolve()
    names = sorted(p.stem for p in factor_dir.glob("alpha*.parquet"))
    if args.factors:
        names = [x.strip() for x in args.factors.split(",") if x.strip()]
    if args.max_factors:
        names = names[: args.max_factors]

    if args.provenance == "base":
        base_summary = pd.read_csv(factor_dir / "point_in_time_run_summary.csv").set_index("factor_name")
        strict_dir = ROOT / "data/processed/alphas_point_in_time_exact_signal_anchor"
        strict_names = {p.stem for p in strict_dir.glob("alpha*.parquet")}
    else:
        base_summary = pd.DataFrame()
        strict_names = set(names)

    rows = []
    curve_rows = []
    daily_rows = []
    failures = []

    for i, name in enumerate(names, 1):
        try:
            panel = pd.read_parquet(factor_dir / f"{name}.parquet")
            if panel.empty or panel.notna().sum().sum() == 0:
                failures.append({"factor": name, "reason": "all_nan"})
                rows.append({"factor": name, "status": "all_nan", "bucket": "unclassified"})
                print(f"[{i}/{len(names)}] {name} all_nan", flush=True)
                continue
            sig = panel.shift(1).reindex(index=qo.index, columns=qo.columns).where(mem)
            direction, pos, neg = choose_direction(sig, fwd, mem, horizons)
            if not np.isfinite(direction):
                failures.append({"factor": name, "reason": "no_valid_direction"})
                rows.append({"factor": name, "status": "all_nan", "bucket": "unclassified"})
                continue
            stats = pos if direction > 0 else neg
            curve = np.array([stats[h].d10 for h in horizons], dtype=float)
            life = lifetime_from_curve(curve)

            # coverage by year (signal only, no forward returns)
            signal_valid = mem & sig.notna()
            cov_years = []
            for y in range(2018, 2023):
                m = mem.loc[f"{y}-01-01":f"{y}-12-31"]
                v = signal_valid.loc[f"{y}-01-01":f"{y}-12-31"]
                cov_years.append(float(v.to_numpy().sum() / m.to_numpy().sum()) if m.to_numpy().sum() else np.nan)

            row = {
                "factor": name,
                "direction": direction,
                "coverage_h1": stats[1].coverage,
                "coverage_min_year": float(np.nanmin(cov_years)) if np.isfinite(cov_years).any() else np.nan,
                "n_days_h1": stats[1].n_days,
                "missing_d10_h1": stats[1].missing_d10,
                "calculation_method": "strict_daily_signal_anchor" if args.provenance == "strict"
                else (base_summary.at[name, "calculation_method"] if name in base_summary.index else "unknown"),
                "is_signal_anchor_exact": True if args.provenance == "strict"
                else (bool(base_summary.at[name, "is_signal_anchor_exact"]) if name in base_summary.index else False),
                "strict_available": True if args.provenance == "strict" else (name in strict_names),
                **life,
            }
            for h in horizons:
                row[f"d10_{h}"] = stats[h].d10
                row[f"d1_{h}"] = stats[h].d1
                row[f"rankic_{h}"] = stats[h].rankic
                row[f"missing_d10_{h}"] = stats[h].missing_d10
                curve_rows.append({
                    "factor": name, "direction": direction, "horizon": h,
                    "d10": stats[h].d10, "d1": stats[h].d1,
                    "rankic": stats[h].rankic, "coverage": stats[h].coverage,
                    "n_days": stats[h].n_days, "missing_d10": stats[h].missing_d10,
                })
                daily_rows.append(pd.DataFrame({
                    "factor": name,
                    "direction": direction,
                    "horizon": h,
                    "date": stats[h].d10_series.index,
                    "d10": stats[h].d10_series.values,
                    "d1": stats[h].d1_series.values,
                    "ew": stats[h].ew_series.values,
                    "missing_d10": stats[h].missing_d10,
                }))
            rows.append(row)
            print(
                f"[{i}/{len(names)}] {name} dir={direction:+.0f} "
                f"peak=D{life['peak_h']} {life['peak_d10']*100:+.1f}% "
                f"lifetime={life['lifetime']} bucket={life['bucket']} "
                f"strict={row['strict_available']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"factor": name, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[{i}/{len(names)}] {name} FAILED: {exc}", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(out / "d10_lifetime.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(out / "d10_curves.csv", index=False)
    if daily_rows:
        pd.concat(daily_rows, ignore_index=True).to_parquet(out / "d10_daily.parquet", index=False)
    pd.DataFrame(failures).to_csv(out / "failures.csv", index=False)

    summary = {
        "period": {"start": "2018-01-01", "end": "2022-12-31"},
        "n_requested": len(names),
        "n_computed": int(len(result)),
        "n_failures": len(failures),
        "bucket_counts": result["bucket"].value_counts().to_dict() if len(result) else {},
        "status_counts": result["status"].value_counts().to_dict() if len(result) else {},
        "strict_available": int(result["strict_available"].sum()) if len(result) else 0,
        "note": "v3 ex-ante D10; ranking uses signal only; EW uses all members with valid forward returns; no 2023+.",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    print("\n完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
