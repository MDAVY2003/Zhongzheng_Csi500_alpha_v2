"""Compute comprehensive performance metrics for saved backtest runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RUNS = {
    "V3_Strict_Bucket4_ND3": "results/v3_portfolio_is_strict/v3_bucket4_nd3",
    "V3_Strict_Equal_ND3": "results/v3_portfolio_is_strict/v3_equal_nd3",
    "V3_Strict_Bucket4_ND5": "results/v3_portfolio_is_strict/v3_bucket4_nd5",
    "V3_Strict_Equal_ND5": "results/v3_portfolio_is_strict/v3_equal_nd5",
}
def max_drawdown(values: pd.Series) -> tuple[float, int]:
    # Include the initial 1.0 so a first-day loss is a real drawdown.
    series = pd.concat(
        [pd.Series([1.0], dtype=float), pd.Series(values, dtype=float).reset_index(drop=True)],
        ignore_index=True,
    )
    peak = series.cummax()
    dd = series / peak - 1.0
    mdd = float(dd.min())
    # longest underwater run in trading days
    underwater = dd < 0
    longest = cur = 0
    for flag in underwater:
        cur = cur + 1 if flag else 0
        longest = max(longest, cur)
    return mdd, int(longest)


def metrics_for(path: Path) -> tuple[dict, pd.Series]:
    daily = pd.read_parquet(path / "daily.parquet")
    bm = pd.read_parquet(path / "benchmarks.parquet")
    df = daily.join(bm, how="inner")
    r = df["net_return"].astype(float)
    g = df["gross_return_before_cost"].astype(float)
    m = df["matched_return"].astype(float)
    n = len(df)
    years = n / 252.0
    nav = (1.0 + r).cumprod()
    cum = float(nav.iloc[-1] - 1.0)
    cagr = float(nav.iloc[-1] ** (252.0 / n) - 1.0)
    vol = float(r.std(ddof=1) * np.sqrt(252))
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if r.std(ddof=1) > 0 else np.nan
    downside = r.where(r < 0, 0.0)
    down_dev = float(np.sqrt((downside ** 2).mean()) * np.sqrt(252))
    sortino = float(r.mean() * 252 / down_dev) if down_dev > 0 else np.nan
    mdd, mdd_days = max_drawdown(nav)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else np.nan
    active = r - m
    te = float(active.std(ddof=1) * np.sqrt(252))
    ir = float(active.mean() / active.std(ddof=1) * np.sqrt(252)) if active.std(ddof=1) > 0 else np.nan
    tstat = float(active.mean() / active.std(ddof=1) * np.sqrt(n)) if active.std(ddof=1) > 0 else np.nan
    beta = float(np.cov(r, m, ddof=1)[0, 1] / np.var(m, ddof=1)) if np.var(m, ddof=1) > 0 else np.nan
    alpha = float((r.mean() - beta * m.mean()) * 252) if np.isfinite(beta) else np.nan
    corr = float(r.corr(m))
    up = m > 0
    down = m < 0
    up_capture = float(r[up].mean() / m[up].mean()) if up.any() and m[up].mean() != 0 else np.nan
    down_capture = float(r[down].mean() / m[down].mean()) if down.any() and m[down].mean() != 0 else np.nan
    gross_cum = float((1.0 + g).prod() - 1.0)
    gross_cagr = float((1.0 + gross_cum) ** (252.0 / n) - 1.0)
    gross_vol = float(g.std(ddof=1) * np.sqrt(252))
    gross_sharpe = float(g.mean() / g.std(ddof=1) * np.sqrt(252)) if g.std(ddof=1) > 0 else np.nan
    monthly = (1.0 + r).resample("ME").prod() - 1.0
    monthly_m = (1.0 + m).resample("ME").prod() - 1.0
    var95 = float(r.quantile(0.05))
    cvar95 = float(r[r <= var95].mean())
    out = {
        "run": path.name,
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "n_days": n,
        "years": years,
        "net_cumulative": cum,
        "cagr": cagr,
        "annualized_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": mdd,
        "max_drawdown_days": mdd_days,
        "daily_win_rate": float((r > 0).mean()),
        "monthly_win_rate": float((monthly > 0).mean()),
        "best_month": float(monthly.max()),
        "worst_month": float(monthly.min()),
        "skew": float(r.skew()),
        "excess_kurtosis": float(r.kurt()),
        "var_95_daily": var95,
        "cvar_95_daily": cvar95,
        "gross_cumulative": gross_cum,
        "gross_cagr": gross_cagr,
        "gross_vol": gross_vol,
        "gross_sharpe": gross_sharpe,
        "cost_drag_cagr": gross_cagr - cagr,
        "matched_cumulative": float((1.0 + m).prod() - 1.0),
        "matched_cagr": float((1.0 + m).prod() ** (252.0 / n) - 1.0),
        "matched_vol": float(m.std(ddof=1) * np.sqrt(252)),
        "matched_sharpe": float(m.mean() / m.std(ddof=1) * np.sqrt(252)) if m.std(ddof=1) > 0 else np.nan,
        "active_cagr": cagr - float((1.0 + m).prod() ** (252.0 / n) - 1.0),
        "tracking_error": te,
        "information_ratio": ir,
        "active_tstat": tstat,
        "beta": beta,
        "annualized_alpha": alpha,
        "correlation": corr,
        "up_capture": up_capture,
        "down_capture": down_capture,
        "total_cost": float(daily["total_cost"].sum()),
        "cost_over_avg_nav_per_year": float(daily["total_cost"].sum() / daily["nav_after_open"].mean() / years),
        "annualized_turnover": float(((daily["stock_buy_notional"] + daily["stock_sell_notional"]) / daily["nav_after_open"].shift(1).fillna(1e7)).mean() * 252),
        "average_stock_exposure": float(daily["stock_exposure"].mean()),
        "mean_stock_count": float(daily["actual_stock_count"].mean()),
    }
    return out, monthly


def pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x*100:.2f}%"


def num(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/performance_metrics")
    parser.add_argument("--runs-root", type=Path, default=None, help="显式读取该组合输出目录的 comparison.csv 所列实验")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"输出目录已存在：{out}")
    out.mkdir(parents=True)
    rows = []
    monthly_frames = {}
    if args.runs_root is None:
        runs = {label: ROOT / rel for label, rel in RUNS.items()}
    else:
        base = args.runs_root.resolve()
        comparison = pd.read_csv(base / "comparison.csv")
        names = comparison["variant"].tolist()
        if not names or comparison["variant"].duplicated().any():
            raise ValueError("组合清单为空或重复")
        runs = {}
        for name in names:
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("组合名称不能包含路径")
            path = base / name
            for required in ["daily.parquet", "benchmarks.parquet"]:
                if not (path / required).is_file():
                    raise FileNotFoundError(f"缺少组合产物：{path / required}")
            runs[name] = path
    for label, path in runs.items():
        if not path.exists():
            print(f"skip missing {path}")
            continue
        row, monthly = metrics_for(path)
        row["label"] = label
        rows.append(row)
        monthly_frames[label] = monthly
    df = pd.DataFrame(rows)
    df.to_csv(out / "performance_summary.csv", index=False)
    for label, s in monthly_frames.items():
        s.to_frame("return").to_csv(out / f"monthly_{label}.csv")
    # main table for report
    cols = ["label", "start", "end", "cagr", "annualized_vol", "sharpe", "sortino", "calmar", "max_drawdown", "information_ratio", "active_tstat", "gross_sharpe", "cost_drag_cagr", "annualized_turnover", "average_stock_exposure"]
    main = df[cols].copy()
    for c in ["cagr", "annualized_vol", "max_drawdown", "cost_drag_cagr"]:
        main[c] = main[c].map(pct)
    for c in ["sharpe", "sortino", "calmar", "information_ratio", "active_tstat", "gross_sharpe", "annualized_turnover", "average_stock_exposure"]:
        main[c] = main[c].map(num)
    (out / "performance_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str) + "\n")
    print(main.to_string(index=False))


if __name__ == "__main__":
    main()
