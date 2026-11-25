"""V3 IS-only: short + middle + delayed in one composite.

Signal:
- short: today's percentile.
- middle: trailing average of daily percentile over `lifetime` days, then re-rank.
- delayed: trailing average over `window_width` days, then re-rank.
- any missing day -> factor score missing; composite missing -> 0.5; all missing -> NaN.

Weights:
- equal-factor: 1/19 each.
- bucket: short 60%, middle 25%, delayed 15%, equal within bucket.

NDROP 3/5.  IS 2018-2022 only; no OOS.
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
    load_panels,
    percentile_panel,
    save_result,
)
from run_short15_ndrop import run_one  # noqa: E402


def trailing_rank(pct: pd.DataFrame, window: int, mem: pd.DataFrame) -> pd.DataFrame:
    if window <= 1:
        return pct
    smooth = pct.rolling(window, min_periods=window).mean().where(mem)
    return smooth.rank(axis=1, method="average", pct=True, na_option="keep")


def weighted_composite(
    scores: dict[str, pd.DataFrame], weights: dict[str, float], mem: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total = None
    valid = None
    for name, score in scores.items():
        w = float(weights[name])
        part = score.fillna(0.5).where(mem) * w
        total = part if total is None else total + part
        count = score.notna().astype("int16")
        valid = count if valid is None else valid + count
    total = total.where(valid.gt(0))
    return total, valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default="2022-12-31")
    args = parser.parse_args()
    if args.end > "2022-12-31":
        raise ValueError("只允许 IS 2018-2022；不得使用 2023+")
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"输出目录已存在：{out}")
    out.mkdir(parents=True)

    sel = pd.read_csv(ROOT / "results/screen_is_v1/selected_factors.csv")
    sel = sel[sel["bucket"].isin(["short", "middle", "delayed"])].copy()
    life = pd.read_csv(ROOT / "results/d10_lifetime_is/d10_lifetime.csv").set_index("factor")
    if len(sel) != 19:
        raise ValueError(f"预期 19 个因子，实际 {len(sel)}")

    panels = load_panels()
    mem = panels["membership"].loc[:"2022-12-31"]
    panels = {k: v.loc[:"2022-12-31"] for k, v in panels.items()}

    pcts = {}
    windows = {}
    for rec in sel.itertuples():
        panel = pd.read_parquet(
            ROOT / f"data/processed/alphas_point_in_time/{rec.factor}.parquet"
        )
        pcts[rec.factor] = percentile_panel(
            panel, rec.direction, panels["qfq_open"], mem
        )
        if rec.bucket == "short":
            windows[rec.factor] = 1
        elif rec.bucket == "middle":
            windows[rec.factor] = int(life.at[rec.factor, "lifetime"])
        else:
            windows[rec.factor] = int(life.at[rec.factor, "lifetime_width"])

    scores = {
        f: trailing_rank(pcts[f], windows[f], mem) for f in pcts
    }

    equal = {f: 1.0 / len(scores) for f in scores}
    bucket_weights = {}
    for bucket, total_w in [("short", 0.60), ("middle", 0.25), ("delayed", 0.15)]:
        names = sel[sel["bucket"].eq(bucket)]["factor"].tolist()
        for f in names:
            bucket_weights[f] = total_w / len(names)

    adv = bt.rolling_valid_observation_mean(panels["amount"], 20)
    cap = bt.rolling_valid_observation_mean(panels["circulating_market_cap"], 20)

    variants = [
        ("all19_equal_nd3", equal, 3),
        ("all19_bucket_nd3", bucket_weights, 3),
        ("all19_equal_nd5", equal, 5),
        ("all19_bucket_nd5", bucket_weights, 5),
    ]

    rows = []
    for name, weights, nd in variants:
        signal, valid = weighted_composite(scores, weights, mem)
        result, cfg = run_one(
            name, signal, panels, adv, cap, args.start, args.end, nd
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
        sharpe = (
            float(daily["net_return"].mean() / daily["net_return"].std(ddof=1) * np.sqrt(252))
            if daily["net_return"].std(ddof=1) > 0
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
        rows.append(
            {
                "variant": name,
                "weighting": "equal" if weights is equal else "bucket_60_25_15",
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
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False, indent=2, default=str), flush=True)

    comp = pd.DataFrame(rows)
    comp.to_csv(out / "comparison.csv", index=False)
    summary = {
        "period": {"start": args.start, "end": args.end},
        "n_factors": len(sel),
        "factor_buckets": sel["bucket"].value_counts().to_dict(),
        "windows": windows,
        "weights": {"equal": equal, "bucket_60_25_15": bucket_weights},
        "comparison": comp.to_dict("records"),
        "note": "仅 IS 2018-2022；无 OOS；缺失因子中性 0.5；滚动窗口严格 min_periods。",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    print("\n完成")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
