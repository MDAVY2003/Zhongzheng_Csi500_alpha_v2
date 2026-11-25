"""IS screening v3 using the ex-ante D10 audit.

Gates: coverage, D10/RankIC sign + 4/5 years, NW t>=2.0, correlation dedupe.
Approximate factors may pass the statistical screen but are excluded from the
final eligible set unless `is_signal_anchor_exact` or a strict file exists.
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
sys.path.insert(0, str(ROOT / "scripts"))

from d10_core import nw_tstat  # noqa: E402
from run_short15_compare import percentile_panel  # noqa: E402


def bucket_horizon(bucket: str, peak_h: float) -> int:
    if bucket == "short":
        return 1
    if bucket == "middle":
        return 3
    if bucket == "long":
        return 6
    if bucket == "delayed":
        return int(peak_h)
    raise ValueError(bucket)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=ROOT / "results/v3_factor_audit")
    parser.add_argument("--strict-audit", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"输出目录已存在：{out}")
    out.mkdir(parents=True)

    life = pd.read_csv(args.audit / "d10_lifetime.csv")
    daily = pd.read_parquet(args.audit / "d10_daily.parquet")
    strict_factors = set()
    if args.strict_audit is not None:
        strict_life = pd.read_csv(args.strict_audit / "d10_lifetime.csv")
        strict_daily = pd.read_parquet(args.strict_audit / "d10_daily.parquet")
        strict_factors = set(strict_life["factor"])
        life = pd.concat([life[~life["factor"].isin(strict_factors)], strict_life], ignore_index=True)
        daily = pd.concat([daily[~daily["factor"].isin(strict_factors)], strict_daily], ignore_index=True)
    data = ROOT / "data/processed/research_market_v1"
    qo = pd.read_parquet(data / "qfq_open.parquet").where(lambda x: x > 0).loc["2018-01-01":"2022-12-31"]
    mem = pd.read_parquet(data / "membership.parquet").astype(bool).loc["2018-01-01":"2022-12-31"]

    rows = []
    panels = {}
    for rec in life.itertuples():
        if rec.status != "success" or not np.isfinite(rec.peak_d10):
            rows.append({"factor": rec.factor, "bucket": rec.bucket, "status": rec.status})
            continue
        h = bucket_horizon(rec.bucket, rec.peak_h)
        d = daily[(daily.factor.eq(rec.factor)) & (daily.horizon.eq(h))].sort_values("date")
        if d.empty:
            rows.append({"factor": rec.factor, "bucket": rec.bucket, "status": "no_daily"})
            continue
        s = d.set_index("date")["d10"]
        t = nw_tstat(s, lag=max(1, h))
        yearly = []
        for y in range(2018, 2023):
            x = s.loc[f"{y}-01-01":f"{y}-12-31"].dropna()
            yearly.append(float(x.mean() * 252 / h) if len(x) else np.nan)
        pos_years = int(sum(v > 0 for v in yearly if np.isfinite(v)))
        d10_h = float(s.mean() * 252 / h)
        rankic_h = float(getattr(rec, f"rankic_{h}", np.nan))
        pass_cov = bool(rec.coverage_h1 >= 0.80 and rec.coverage_min_year >= 0.60)
        pass_long = bool(d10_h > 0 and rankic_h > 0 and pos_years >= 4)
        pass_t = bool(np.isfinite(t) and t >= 2.0)
        exact_flag = bool(rec.is_signal_anchor_exact) if pd.notna(rec.is_signal_anchor_exact) else False
        strict_flag = bool(rec.strict_available) if pd.notna(rec.strict_available) else False
        final_eligible = exact_flag or strict_flag
        rows.append({
            "factor": rec.factor,
            "bucket": rec.bucket,
            "direction": rec.direction,
            "peak_h": rec.peak_h,
            "screen_h": h,
            "d10_h": d10_h,
            "rankic_h": rankic_h,
            "nw_t": t,
            "d10_positive_years": pos_years,
            "coverage_avg": rec.coverage_h1,
            "coverage_min_year": rec.coverage_min_year,
            "is_signal_anchor_exact": exact_flag,
            "strict_available": strict_flag,
            "final_eligible": final_eligible,
            "pass_coverage": pass_cov,
            "pass_long_leg": pass_long,
            "pass_t": pass_t,
            "pass_pre_dedupe": bool(pass_cov and pass_long and pass_t),
            "status": "ok",
        })
        if pass_cov and pass_long and pass_t:
            source_dir = (
                ROOT / "data/processed/alphas_point_in_time_strict_v3"
                if rec.factor in strict_factors
                else ROOT / "data/processed/alphas_point_in_time"
            )
            panel = pd.read_parquet(source_dir / f"{rec.factor}.parquet")
            panels[rec.factor] = percentile_panel(panel, rec.direction, qo, mem)

    all_df = pd.DataFrame(rows).sort_values("d10_h", ascending=False, na_position="last")
    all_df.to_csv(out / "screen_all.csv", index=False)
    for col in ["final_eligible", "pass_pre_dedupe", "pass_coverage", "pass_long_leg", "pass_t"]:
        all_df[col] = all_df[col].fillna(False).astype(bool)
    passed = all_df[all_df["pass_pre_dedupe"]].copy()
    passed.to_csv(out / "pass_pre_dedupe.csv", index=False)

    eligible = passed[passed["final_eligible"]].copy()
    corr = pd.DataFrame()
    selected = eligible.copy()
    if len(eligible) > 1:
        factors = eligible["factor"].tolist()
        corr = pd.DataFrame(np.nan, index=factors, columns=factors, dtype=float)
        for a in factors:
            corr.loc[a, a] = 1.0
            for b in factors:
                if a >= b:
                    continue
                arr = panels[a].corrwith(panels[b], axis=1).dropna()
                v = float(arr.mean()) if len(arr) else np.nan
                corr.loc[a, b] = v
                corr.loc[b, a] = v
        corr.to_csv(out / "pass_correlation.csv")
        kept = []
        for rec in eligible.sort_values("d10_h", ascending=False).itertuples():
            if not kept:
                kept.append(rec.factor)
                continue
            mx = max(abs(corr.loc[rec.factor, k]) for k in kept)
            if np.isfinite(mx) and mx > 0.70:
                continue
            kept.append(rec.factor)
        selected = eligible[eligible["factor"].isin(kept)].sort_values("d10_h", ascending=False)
    selected.to_csv(out / "selected_factors.csv", index=False)
    approx_only = passed[~passed["final_eligible"].astype(bool)]
    approx_only.to_csv(out / "approx_only_excluded.csv", index=False)

    summary = {
        "period": {"start": "2018-01-01", "end": "2022-12-31"},
        "n_audited": int(len(life)),
        "n_pass_pre_dedupe": int(len(passed)),
        "n_eligible_pre_dedupe": int(len(eligible)),
        "n_selected": int(len(selected)),
        "n_approx_only_excluded": int(len(approx_only)),
        "bucket_counts_selected": selected["bucket"].value_counts().to_dict() if len(selected) else {},
        "gates": {"coverage_avg": 0.80, "coverage_year": 0.60, "positive_years": 4, "nw_t": 2.0, "corr": 0.70},
        "horizons": {"short": 1, "middle": 3, "long": 6, "delayed": "peak_h"},
        "note": "IS only; ex-ante D10; final eligible requires exact flag or strict file.",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
