"""Static frozen OOS: all 15 short factors, lifetime-smoothed, NDROP 3/5.

OOS 2023-2025.  Factors/directions/lifetimes are frozen from IS 2018-2022.
No tuning after seeing the result.
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

import realistic_backtest as bt  # noqa: E402
from run_short15_compare import (  # noqa: E402
    benchmark,
    composite,
    load_panels,
    percentile_panel,
    save_result,
)
from run_short15_ndrop import run_one  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=str, default="2023-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--ndrops", type=str, default="3,5")
    parser.add_argument("--allow-oos", action="store_true")
    args = parser.parse_args()
    if not args.allow_oos:
        raise ValueError("OOS 需要显式 --allow-oos")
    if args.start < "2023-01-01" or args.end > "2025-12-31":
        raise ValueError("本实验只允许 2023-01-01 至 2025-12-31")
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"输出目录已存在：{out}")
    out.mkdir(parents=True)

    sel = pd.read_csv(ROOT / "results/screen_is_v1/selected_factors.csv")
    sel = sel[sel["bucket"].eq("short")].copy()
    if len(sel) != 15:
        raise ValueError(f"预期 15 个 short 因子，实际 {len(sel)}")
    life = pd.read_csv(ROOT / "results/d10_lifetime_is/d10_lifetime.csv").set_index("factor")
    lifetimes = {r.factor: int(life.at[r.factor, "lifetime"]) for r in sel.itertuples()}

    panels = load_panels()
    mem = panels["membership"].loc[:"2025-12-31"]
    panels = {k: v.loc[:"2025-12-31"] for k, v in panels.items()}

    percentiles = {}
    for rec in sel.itertuples():
        panel = pd.read_parquet(
            ROOT / f"data/processed/alphas_point_in_time/{rec.factor}.parquet"
        )
        percentiles[rec.factor] = percentile_panel(
            panel, rec.direction, panels["qfq_open"], mem
        )
    smoothed, valid = composite(percentiles, lifetimes, mem)

    adv = bt.rolling_valid_observation_mean(panels["amount"], 20)
    cap = bt.rolling_valid_observation_mean(panels["circulating_market_cap"], 20)

    ndrops = [int(x.strip()) for x in args.ndrops.split(",") if x.strip()]
    rows = []
    for nd in ndrops:
        name = f"short15_smoothed_ndrop{nd}"
        result, cfg = run_one(
            name, smoothed, panels, adv, cap, args.start, args.end, nd
        )
        metric = save_result(result, out / name, cfg.initial_capital)
        daily = result["daily"]
        matched = benchmark(daily, result["constituent_exits"], cfg.initial_capital)
        active = daily["net_return"] - matched
        net_cum = float(daily["net_value"].iloc[-1] - 1.0)
        matched_cum = float((1.0 + matched).prod() - 1.0)
        ir = (
            float(active.mean() / active.std(ddof=1) * np.sqrt(252))
            if active.std(ddof=1) > 0
            else np.nan
        )
        tstat = (
            float(active.mean() / active.std(ddof=1) * np.sqrt(len(active)))
            if active.std(ddof=1) > 0
            else np.nan
        )
        pd.DataFrame(
            {
                "strategy_net_return": daily["net_return"],
                "strategy_net_value": daily["net_value"],
                "matched_return": matched,
                "matched_net_value": (1.0 + matched).cumprod(),
                "active_return": active,
            }
        ).to_parquet(out / name / "benchmarks.parquet")
        row = {
            "variant": name,
            "ndrop": nd,
            "start": str(daily.index.min().date()),
            "end": str(daily.index.max().date()),
            "net_cumulative": net_cum,
            "gross_cumulative_return": metric["gross_cumulative_return"],
            "matched_cumulative": matched_cum,
            "net_minus_matched": net_cum - matched_cum,
            "net_ir": ir,
            "net_alpha_tstat": tstat,
            "total_cost": metric["total_cost"],
            "annualized_turnover": metric["annualized_turnover"],
            "max_drawdown": metric["max_drawdown"],
            "average_stock_exposure": metric["average_stock_exposure"],
            "mean_stock_count": metric["mean_stock_count"],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2, default=str), flush=True)

    comp = pd.DataFrame(rows)
    comp.to_csv(out / "comparison.csv", index=False)
    summary = {
        "period": {"start": args.start, "end": args.end},
        "n_factors": len(sel),
        "ndrops": ndrops,
        "factors": sel[["factor", "direction"]].to_dict("records"),
        "lifetimes": lifetimes,
        "comparison": comp.to_dict("records"),
        "note": "一次性 OOS；全 15 short；寿命平滑；跑完不调参。",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    print("\n完成")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
