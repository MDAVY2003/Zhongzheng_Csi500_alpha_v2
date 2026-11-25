"""Short-15 comparison: original daily percentile vs lifetime-smoothed percentile.

Only IS 2018-2022 is used.  The 15 short factors and directions are read from
`results/screen_is_v1/selected_factors.csv`; lifetimes from
`results/d10_lifetime_is/d10_lifetime.csv`.

Original:
    15 factors, each uses today's cross-sectional percentile.

Lifetime-smoothed:
    lifetime == 1: today's percentile.
    lifetime == 2: average today's and previous trading day's percentile for
    the same stock; if either is missing, the smoothed score is missing; then
    re-rank cross-sectionally today.

Both:
    equal 1/15 weights, missing factor -> neutral 0.5; all 15 missing -> NaN.
    80 names, 80% exposure, 75-85% band, 20% cash, 1.5% single-stock cap.
    entry 15%, retention 25%; no NDrop; next-open execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import realistic_backtest as bt  # noqa: E402
from research_benchmark import close_exit_matched_returns  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_panels() -> dict[str, pd.DataFrame]:
    cache = ROOT / "data/processed/research_market_v1"
    panels = {p.stem: pd.read_parquet(p) for p in cache.glob("*.parquet")}
    mem = panels["membership"]
    for key, frame in panels.items():
        if not frame.index.equals(mem.index) or not frame.columns.equals(mem.columns):
            raise ValueError(f"panel axes differ: {key}")
    return panels


def percentile_panel(
    panel: pd.DataFrame, direction: float, qo: pd.DataFrame, mem: pd.DataFrame
) -> pd.DataFrame:
    raw = panel.shift(1).reindex(index=qo.index, columns=qo.columns) * float(direction)
    raw = raw.where(mem)
    return raw.rank(axis=1, method="average", pct=True, na_option="keep")


def composite(
    percentiles: dict[str, pd.DataFrame], lifetimes: dict[str, int], mem: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = []
    valid_count = None
    for name, pct in percentiles.items():
        score = pct
        if lifetimes[name] == 2:
            prev = pct.shift(1)
            smooth = (pct + prev) / 2.0
            smooth = smooth.where(mem)
            score = smooth.rank(axis=1, method="average", pct=True, na_option="keep")
        values.append(score)
        count = score.notna().astype("int16")
        valid_count = count if valid_count is None else valid_count + count
    filled = [v.fillna(0.5).where(mem) for v in values]
    comp = sum(filled) / len(filled)
    comp = comp.where(valid_count.gt(0))
    return comp, valid_count


def save_result(result: dict, dest: Path, initial_capital: float) -> dict:
    dest.mkdir(parents=True, exist_ok=False)
    for key in ["daily", "trades", "holdings", "orders", "constituent_exits"]:
        frame = result.get(key)
        if isinstance(frame, pd.DataFrame):
            frame.to_parquet(dest / f"{key}.parquet")
    daily = result["daily"]
    daily.to_parquet(dest / "daily.parquet")
    metric = bt._summary(daily["net_return"], daily["nav_after_open"]).to_dict()
    metric["gross_cumulative_return"] = float(
        (1.0 + daily["gross_return_before_cost"].fillna(0.0)).prod() - 1.0
    )
    metric["total_cost"] = float(daily["total_cost"].sum())
    metric["annualized_turnover"] = float(
        (
            (daily["stock_buy_notional"] + daily["stock_sell_notional"])
            / daily["nav_after_open"].shift(1, fill_value=initial_capital)
        ).mean()
        * 252
    )
    metric["average_stock_exposure"] = float(daily["stock_exposure"].mean())
    metric["max_stock_count"] = int(daily["actual_stock_count"].max())
    metric["mean_stock_count"] = float(daily["actual_stock_count"].mean())
    pd.DataFrame([metric]).to_csv(dest / "performance.csv", index=False)
    bt._yearly_summary(daily).to_csv(dest / "yearly.csv")
    return metric


def run_one(
    name: str,
    signal: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    adv: pd.DataFrame,
    cap: pd.DataFrame,
    start: str,
    end: str,
) -> tuple[dict, dict]:
    cfg = bt.BacktestConfig(
        strategy_mode="buffered_v2",
        initial_capital=10_000_000,
        target_stock_exposure=0.80,
        target_stock_count=80,
        entry_fraction=0.15,
        retention_fraction=0.25,
        stock_exposure_lower=0.75,
        stock_exposure_upper=0.85,
        single_stock_weight_cap=0.015,
        idealized_constituent_exit=True,
    )
    cfg.validate()
    result = bt.backtest_single_factor(
        signal,
        panels["qfq_open"],
        panels["raw_open"],
        panels["amount"],
        panels["circulating_market_cap"],
        panels["up_limit"],
        panels["down_limit"],
        panels["full_day_suspension"],
        panels["membership"],
        None,
        None,
        None,
        factor_name=name,
        selected_direction=1.0,
        start_date=start,
        end_date=end,
        config=cfg,
        qfq_close=panels["qfq_close"],
        raw_close=panels["raw_close"],
        precomputed_stock_adv=adv,
        precomputed_market_cap_avg=cap,
        factor_signal_lag=0,
        futures_hedge_enabled=False,
        independent_stock_sets=True,
        short_factor=None,
        short_veto_config=None,
        max_signal_replacements_per_day=None,
    )
    return result, cfg


def benchmark(daily: pd.DataFrame, exits: pd.DataFrame, initial_capital: float) -> pd.Series:
    spot = pd.read_parquet(ROOT / "data/raw/index_daily/csi500.parquet")
    spot.index = pd.to_datetime(spot["trade_date"], format="%Y%m%d")
    spot = spot.reindex(daily.index)
    return close_exit_matched_returns(
        daily, spot["open"], spot["close"], exits, initial_capital=initial_capital
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default="2022-12-31")
    args = parser.parse_args()
    if args.end > "2022-12-31":
        raise ValueError("本实验只允许 IS 2018-2022；不得使用 2023+")
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
    if not all(v in (1, 2) for v in lifetimes.values()):
        raise ValueError(f"short 因子寿命应为 1/2：{lifetimes}")

    panels = load_panels()
    # Include pre-2018 dates so the engine has a prior signal date; never go past 2022-12-31.
    qo = panels["qfq_open"].loc[:"2022-12-31"]
    mem = panels["membership"].loc[:"2022-12-31"]
    panels = {k: v.loc[:"2022-12-31"] for k, v in panels.items()}

    percentiles = {}
    for rec in sel.itertuples():
        panel = pd.read_parquet(
            ROOT / f"data/processed/alphas_point_in_time/{rec.factor}.parquet"
        )
        percentiles[rec.factor] = percentile_panel(
            panel, rec.direction, panels["qfq_open"], mem
        )

    original, valid_original = composite(
        percentiles, {f: 1 for f in percentiles}, mem
    )
    smoothed, valid_smoothed = composite(percentiles, lifetimes, mem)

    adv = bt.rolling_valid_observation_mean(panels["amount"], 20)
    cap = bt.rolling_valid_observation_mean(panels["circulating_market_cap"], 20)

    rows = []
    for name, signal, valid in [
        ("original", original, valid_original),
        ("lifetime_smoothed", smoothed, valid_smoothed),
    ]:
        result, cfg = run_one(
            name, signal, panels, adv, cap, args.start, args.end
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
            "start": str(daily.index.min().date()),
            "end": str(daily.index.max().date()),
            "net_cumulative": net_cum,
            "matched_cumulative": matched_cum,
            "net_minus_matched": net_cum - matched_cum,
            "net_ir": ir,
            "net_alpha_tstat": tstat,
            "total_cost": metric["total_cost"],
            "annualized_turnover": metric["annualized_turnover"],
            "max_drawdown": metric["max_drawdown"],
            "average_stock_exposure": metric["average_stock_exposure"],
            "mean_stock_count": metric["mean_stock_count"],
            "gross_cumulative_return": metric["gross_cumulative_return"],
            "signal_coverage_mean": float(
                (valid.gt(0).sum(axis=1) / mem.sum(axis=1)).mean()
            ),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2, default=str), flush=True)

    comp = pd.DataFrame(rows)
    comp.to_csv(out / "comparison.csv", index=False)
    summary = {
        "period": {"start": args.start, "end": args.end},
        "factors": sel[["factor", "direction"]].to_dict("records"),
        "lifetimes": lifetimes,
        "n_factors": len(sel),
        "n_lifetime_1": sum(v == 1 for v in lifetimes.values()),
        "n_lifetime_2": sum(v == 2 for v in lifetimes.values()),
        "comparison": comp.to_dict("records"),
        "note": "仅 IS 2018-2022；无 NDrop；80 只/80% 仓位；等权 1/15；缺失因子中性 0.5。",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    (out / "config_snapshot.json").write_text(
        json.dumps(
            {
                "engine": bt.BacktestConfig(
                    strategy_mode="buffered_v2",
                    initial_capital=10_000_000,
                    target_stock_exposure=0.80,
                    target_stock_count=80,
                    entry_fraction=0.15,
                    retention_fraction=0.25,
                    stock_exposure_lower=0.75,
                    stock_exposure_upper=0.85,
                    single_stock_weight_cap=0.015,
                    idealized_constituent_exit=True,
                ).__dict__,
                "command": sys.argv,
                "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n"
    )
    (out / "run_metadata.json").write_text(
        json.dumps(
            {
                "data_sha256": {
                    "membership": sha256(ROOT / "data/processed/research_market_v1/membership.parquet"),
                    "qfq_open": sha256(ROOT / "data/processed/research_market_v1/qfq_open.parquet"),
                },
                "code_sha256": {
                    str(p.relative_to(ROOT)): sha256(p)
                    for p in [Path(__file__), ROOT / "src/realistic_backtest.py", ROOT / "src/research_benchmark.py"]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print("\n完成")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
