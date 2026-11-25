"""V3 IS portfolio: selected exact factors, lifetime-smoothed, equal/bucket, NDROP 3/5."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import realistic_backtest as bt  # noqa: E402
from v3_factor_sources import resolve_portfolio_sources  # noqa: E402
from run_short15_compare import benchmark, load_panels, percentile_panel, save_result  # noqa: E402
from run_short15_ndrop import run_one  # noqa: E402
from run_v3_mid_delayed_is import trailing_rank, weighted_composite  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screen", type=Path, default=ROOT / "results/v3_screen/selected_factors.csv")
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default="2022-12-31")
    parser.add_argument("--base-audit", type=Path, default=ROOT / "results/v3_factor_audit/d10_lifetime.csv")
    parser.add_argument("--strict-audit", type=Path, default=ROOT / "results/v3_strict_audit/d10_lifetime.csv")
    parser.add_argument("--base-factor-dir", type=Path, default=ROOT / "data/processed/alphas_point_in_time")
    parser.add_argument("--strict-factor-dir", type=Path, default=ROOT / "data/processed/alphas_point_in_time_strict_v3")
    args = parser.parse_args()
    if args.end > "2022-12-31":
        raise ValueError("只允许 IS 2018-2022")
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"输出目录已存在：{out}")
    out.mkdir(parents=True)

    sel = pd.read_csv(args.screen)
    if len(sel) == 0:
        raise ValueError("没有入选因子")
    if not bool(sel["final_eligible"].all()):
        raise ValueError("入选因子中存在非 exact 因子")
    sources = resolve_portfolio_sources(
        sel, base_audit=args.base_audit, strict_audit=args.strict_audit,
        base_factor_dir=args.base_factor_dir, strict_factor_dir=args.strict_factor_dir,
    )

    panels = load_panels()
    mem = panels["membership"].loc[:"2022-12-31"]
    panels = {k: v.loc[:"2022-12-31"] for k, v in panels.items()}
    pcts = {}
    windows = {}
    for name, source in sources.items():
        panel = pd.read_parquet(source.path)
        pcts[name] = percentile_panel(panel, source.direction, panels["qfq_open"], mem)
        windows[name] = source.window
    scores = {f: trailing_rank(pcts[f], windows[f], mem) for f in pcts}
    equal = {f: 1.0 / len(scores) for f in scores}
    bucket = {}
    for b, total in [("short", 0.55), ("middle", 0.20), ("long", 0.10), ("delayed", 0.15)]:
        names = sel[sel["bucket"].eq(b)]["factor"].tolist()
        if names:
            for f in names:
                bucket[f] = total / len(names)
    # if a bucket is absent, its weight stays unused; caller should note it.

    adv = bt.rolling_valid_observation_mean(panels["amount"], 20)
    cap = bt.rolling_valid_observation_mean(panels["circulating_market_cap"], 20)

    variants = [
        ("v3_equal_nd3", equal, 3),
        ("v3_bucket4_nd3", bucket, 3),
        ("v3_equal_nd5", equal, 5),
        ("v3_bucket4_nd5", bucket, 5),
    ]
    rows = []
    for name, weights, nd in variants:
        signal, valid = weighted_composite(scores, weights, mem)
        result, cfg = run_one(name, signal, panels, adv, cap, args.start, args.end, nd)
        metric = save_result(result, out / name, cfg.initial_capital)
        (out / name / "config_snapshot.json").write_text(json.dumps({"backtest_config": asdict(cfg), "ndrop": nd, "weights": weights, "windows": windows}, ensure_ascii=False, indent=2) + "\n")
        daily = result["daily"]
        matched = benchmark(daily, result["constituent_exits"], cfg.initial_capital)
        active = daily["net_return"] - matched
        net_cum = float(daily["net_value"].iloc[-1] - 1.0)
        matched_cum = float((1.0 + matched).prod() - 1.0)
        ir = float(active.mean() / active.std(ddof=1) * np.sqrt(252)) if active.std(ddof=1) > 0 else np.nan
        tstat = float(active.mean() / active.std(ddof=1) * np.sqrt(len(active))) if active.std(ddof=1) > 0 else np.nan
        sharpe = float(daily["net_return"].mean() / daily["net_return"].std(ddof=1) * np.sqrt(252)) if daily["net_return"].std(ddof=1) > 0 else np.nan
        pd.DataFrame({
            "strategy_net_return": daily["net_return"],
            "strategy_net_value": daily["net_value"],
            "matched_return": matched,
            "matched_net_value": (1.0 + matched).cumprod(),
            "active_return": active,
        }).to_parquet(out / name / "benchmarks.parquet")
        rows.append({
            "variant": name,
            "weighting": "equal" if weights is equal else "bucket_55_20_10_15",
            "ndrop": nd,
            "start": str(daily.index.min().date()),
            "end": str(daily.index.max().date()),
            "net_cumulative": net_cum,
            "gross_cumulative_return": metric["gross_cumulative_return"],
            "matched_cumulative": matched_cum,
            "net_minus_matched": net_cum - matched_cum,
            "net_ir": ir,
            "net_alpha_tstat": tstat,
            "sharpe": sharpe,
            "total_cost": metric["total_cost"],
            "annualized_turnover": metric["annualized_turnover"],
            "max_drawdown": metric["max_drawdown"],
            "average_stock_exposure": metric["average_stock_exposure"],
            "mean_stock_count": metric["mean_stock_count"],
        })
        print(json.dumps(rows[-1], ensure_ascii=False, indent=2, default=str), flush=True)

    comp = pd.DataFrame(rows)
    comp.to_csv(out / "comparison.csv", index=False)
    summary = {
        "factor_sources": {name: {"path": str(source.path), "audit_path": str(source.audit_path),
            "provenance": source.provenance, "direction": source.direction,
            "bucket": source.bucket, "window": source.window} for name, source in sources.items()},
        "period": {"start": args.start, "end": args.end},
        "n_factors": len(sel),
        "bucket_counts": sel["bucket"].value_counts().to_dict(),
        "windows": windows,
        "weights_equal": equal,
        "weights_bucket": bucket,
        "comparison": comp.to_dict("records"),
        "note": "仅 IS；exact/strict 入选因子；short 1 日，middle/long 按 lifetime，delayed 按 width；四桶权重 55-20-10-15。",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
