"""IS-only factor screening (v1) using the agreed 4 gates.

Gates:
1. coverage: average >= 80%, every IS year >= 60%;
2. long leg at the bucket horizon: D10 > 0, RankIC > 0, D10 positive in >= 4/5 years;
3. Newey-West t of the D10 daily series >= 2.0;
4. correlation dedupe at 0.70 after the first three gates.

No 2023-2025 data is used. Lifetime-stability is diagnostic only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def nw_tstat(series: pd.Series, lag: int) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) < max(20, lag + 2):
        return np.nan
    e = (x - x.mean()).to_numpy(dtype=float)
    t = len(e)
    gamma0 = float(np.mean(e * e))
    var = gamma0
    for ell in range(1, lag + 1):
        gamma = float(np.mean(e[ell:] * e[:-ell]))
        var += 2.0 * (1.0 - ell / (lag + 1.0)) * gamma
    if not np.isfinite(var) or var <= 0:
        return np.nan
    return float(x.mean() / math.sqrt(var / t))


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
    parser.add_argument(
        "--lifetime", type=Path, default=ROOT / "results/d10_lifetime_is/d10_lifetime.csv"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"输出目录已存在：{out}")
    out.mkdir(parents=True)

    data = ROOT / "data/processed/research_market_v1"
    qo = pd.read_parquet(data / "qfq_open.parquet").where(lambda x: x > 0)
    mem = pd.read_parquet(data / "membership.parquet").astype(bool)
    start, end = "2018-01-01", "2022-12-31"
    qo = qo.loc[start:end]
    mem = mem.loc[start:end]

    life = pd.read_csv(args.lifetime)
    factor_dir = ROOT / "data/processed/alphas_point_in_time"

    rows = []
    daily_store = {}
    for i, rec in enumerate(life.itertuples(), 1):
        name = rec.factor
        if rec.status != "success" or not np.isfinite(rec.peak_d10):
            rows.append(
                {
                    "factor": name,
                    "bucket": rec.bucket,
                    "direction": rec.direction,
                    "peak_h": rec.peak_h,
                    "peak_d10": rec.peak_d10,
                    "coverage_avg": rec.coverage_h1,
                    "coverage_min_year": np.nan,
                    "screen_h": np.nan,
                    "d10_h": np.nan,
                    "rankic_h": np.nan,
                    "d10_positive_years": np.nan,
                    "nw_t": np.nan,
                    "pass_coverage": False,
                    "pass_long_leg": False,
                    "pass_t": False,
                    "pass_pre_dedupe": False,
                    "screen_status": rec.status,
                }
            )
            continue

        panel = pd.read_parquet(factor_dir / f"{name}.parquet")
        sig = panel.shift(1).reindex(index=qo.index, columns=qo.columns).where(mem)
        sig = sig * float(rec.direction)

        # coverage (signal availability, not forward labels)
        valid_sig = sig.notna() & mem
        cov_avg = float(valid_sig.to_numpy().sum() / mem.to_numpy().sum())
        yearly_cov = []
        for y in range(2018, 2023):
            m = mem.loc[f"{y}-01-01":f"{y}-12-31"]
            v = valid_sig.loc[f"{y}-01-01":f"{y}-12-31"]
            yearly_cov.append(float(v.to_numpy().sum() / m.to_numpy().sum()))
        cov_min = float(np.min(yearly_cov))

        h = bucket_horizon(rec.bucket, rec.peak_h)
        fwd = qo.shift(-h) / qo - 1.0
        valid = sig.notna() & fwd.notna() & mem
        pct = sig.where(valid).rank(axis=1, method="first", pct=True)
        dec = np.ceil(pct * 10).clip(1, 10)
        ew = fwd.where(valid).mean(axis=1)
        ex = fwd.where(valid).sub(ew, axis=0)
        d10_series = ex.where(dec.eq(10)).mean(axis=1)
        d10_h = float(d10_series.mean() * 252 / h)
        nw = nw_tstat(d10_series, lag=max(1, h))

        ic = (
            sig.where(valid).rank(axis=1).corrwith(fwd.where(valid).rank(axis=1), axis=1)
        ).dropna()
        rankic_h = float(ic.mean()) if len(ic) else np.nan

        yearly_d10 = []
        for y in range(2018, 2023):
            x = d10_series.loc[f"{y}-01-01":f"{y}-12-31"].dropna()
            yearly_d10.append(float(x.mean() * 252 / h) if len(x) else np.nan)
        pos_years = int(sum(v > 0 for v in yearly_d10 if np.isfinite(v)))

        pass_cov = bool(cov_avg >= 0.80 and cov_min >= 0.60)
        pass_long = bool(
            np.isfinite(d10_h)
            and np.isfinite(rankic_h)
            and d10_h > 0
            and rankic_h > 0
            and pos_years >= 4
        )
        pass_t = bool(np.isfinite(nw) and nw >= 2.0)
        pass_pre = bool(pass_cov and pass_long and pass_t)

        rows.append(
            {
                "factor": name,
                "bucket": rec.bucket,
                "direction": rec.direction,
                "peak_h": rec.peak_h,
                "peak_d10": rec.peak_d10,
                "coverage_avg": cov_avg,
                "coverage_min_year": cov_min,
                "screen_h": h,
                "d10_h": d10_h,
                "rankic_h": rankic_h,
                "d10_positive_years": pos_years,
                "nw_t": nw,
                "pass_coverage": pass_cov,
                "pass_long_leg": pass_long,
                "pass_t": pass_t,
                "pass_pre_dedupe": pass_pre,
                "screen_status": "ok",
            }
        )
        if pass_pre:
            daily_store[name] = sig.where(mem)

        print(
            f"[{i}/{len(life)}] {name} {rec.bucket} h={h} "
            f"D10={d10_h*100:+.1f}% IC={rankic_h:+.4f} "
            f"years={pos_years}/5 t={nw:.2f} pass={pass_pre}",
            flush=True,
        )

    result = pd.DataFrame(rows)
    result.to_csv(out / "screen_all.csv", index=False)
    passed = result[result["pass_pre_dedupe"]].copy()
    passed = passed.sort_values("d10_h", ascending=False).reset_index(drop=True)
    passed.to_csv(out / "pass_pre_dedupe.csv", index=False)

    selected = passed.copy()
    corr_matrix = pd.DataFrame()
    if len(passed) > 1:
        factors = passed["factor"].tolist()
        corr_matrix = pd.DataFrame(np.nan, index=factors, columns=factors, dtype=float)
        for a in factors:
            corr_matrix.loc[a, a] = 1.0
            for b in factors:
                if a >= b:
                    continue
                arr = daily_store[a].loc[start:end].corrwith(
                    daily_store[b].loc[start:end], axis=1
                ).dropna()
                val = float(arr.mean()) if len(arr) else np.nan
                corr_matrix.loc[a, b] = val
                corr_matrix.loc[b, a] = val
        corr_matrix.to_csv(out / "pass_correlation.csv")

        kept = []
        for rec in passed.itertuples():
            if not kept:
                kept.append(rec.factor)
                continue
            max_corr = max(abs(corr_matrix.loc[rec.factor, other]) for other in kept)
            if np.isfinite(max_corr) and max_corr > 0.70:
                continue
            kept.append(rec.factor)
        selected = passed[passed["factor"].isin(kept)].copy()
        selected["max_corr_with_kept"] = [
            max([abs(corr_matrix.loc[f, k]) for k in kept if k != f], default=np.nan)
            for f in selected["factor"]
        ]
    selected.to_csv(out / "selected_factors.csv", index=False)

    summary = {
        "period": {"start": start, "end": end},
        "n_factors": int(len(result)),
        "n_pass_pre_dedupe": int(len(passed)),
        "n_selected": int(len(selected)),
        "bucket_counts_all": result["bucket"].value_counts().to_dict(),
        "bucket_counts_selected": selected["bucket"].value_counts().to_dict()
        if len(selected)
        else {},
        "gates": {
            "coverage_avg_min": 0.80,
            "coverage_year_min": 0.60,
            "d10_positive_years_min": 4,
            "nw_t_min": 2.0,
            "corr_max": 0.70,
        },
        "horizons": {"short": 1, "middle": 3, "long": 6, "delayed": "peak_h"},
        "selected": selected[
            [
                "factor",
                "bucket",
                "direction",
                "screen_h",
                "d10_h",
                "rankic_h",
                "nw_t",
                "d10_positive_years",
                "coverage_avg",
            ]
        ].to_dict("records")
        if len(selected)
        else [],
        "note": "仅 IS 2018-2022；2023-2025 未使用；寿命稳定性仅诊断，不筛选。",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    print("\n完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
