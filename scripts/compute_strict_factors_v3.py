"""Compute strict signal-date anchored values for selected approximate factors.

Generic re-anchoring wrapper:
- raw cumulative price = qfq price * last adj_factor per stock;
- for each signal date, divide the lookback window by adj_factor at the signal date;
- run the registered alpha function on that window with the CSI500 universe mask;
- take the last row.

This is slower than the vectorized base engine but exact for the selected factors.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import alphas  # noqa: E402
import operators as op  # noqa: E402
from alpha_registry import select_alpha_specs  # noqa: E402
from factor_data import FactorData, load_sw_industry_panels  # noqa: E402

STRICT_FACTORS = ["alpha001", "alpha013", "alpha015", "alpha058", "alpha068",
                  "alpha075", "alpha089", "alpha092", "alpha098"]


def load_all() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    bars = pd.read_parquet(
        ROOT / "data/processed/daily_bars.parquet",
        columns=["date", "code", "open", "high", "low", "close", "volume", "amount", "vwap", "returns"],
    )
    bars["date"] = pd.to_datetime(bars["date"], errors="raise")
    bars["code"] = bars["code"].astype(str).str.upper()
    bars = bars.sort_values(["date", "code"])
    panels = {
        f: bars.pivot(index="date", columns="code", values=f).sort_index()
        for f in ["open", "high", "low", "close", "volume", "amount", "vwap", "returns"]
    }
    dates = panels["close"].index
    codes = panels["close"].columns

    adjs = {}
    endpoints = {}
    for f in sorted((ROOT / "data/raw/daily").glob("*.parquet")):
        d = pd.read_parquet(f, columns=["date", "code", "adj_factor"])
        if d.empty:
            continue
        d["date"] = pd.to_datetime(d["date"])
        code = str(d["code"].iloc[0]).upper()
        endpoints[code] = float(d.sort_values("date")["adj_factor"].iloc[-1])
        adjs[code] = d.set_index("date")["adj_factor"]
    adj = pd.DataFrame(adjs).sort_index().reindex(index=dates, columns=codes)
    endpoint = pd.Series(endpoints).reindex(codes)
    for f in ["open", "high", "low", "close", "vwap"]:
        panels[f] = panels[f] * endpoint
    industry = load_sw_industry_panels(
        ROOT / "data/raw/factor_dependencies/sw_industry_membership.parquet",
        dates=dates,
        codes=codes,
    )
    return panels, adj, endpoint, industry


def load_universe(dates: pd.DatetimeIndex, codes: pd.Index) -> pd.DataFrame:
    m = pd.read_parquet(ROOT / "data/processed/universe/csi500_daily_membership.parquet")
    m["date"] = pd.to_datetime(m["date"])
    m["code"] = m["code"].astype(str).str.upper()
    return (
        m.pivot(index="date", columns="code", values="is_member")
        .reindex(index=dates, columns=codes, fill_value=False)
        .fillna(False)
        .astype(bool)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factors", type=str, default=",".join(STRICT_FACTORS))
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default="2022-12-31")
    parser.add_argument("--lookback", type=int, default=150)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"输出目录已存在：{out}")
    out.mkdir(parents=True)

    factors = [x.strip() for x in args.factors.split(",") if x.strip()]
    specs = {s.function_name: s for s in select_alpha_specs()}
    panels, adj, endpoint, industry = load_all()
    dates = panels["close"].index
    codes = panels["close"].columns
    universe = load_universe(dates, codes)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    exec_dates = dates[(dates >= start) & (dates <= end)]
    summary = []
    for name in factors:
        spec = specs[name]
        fn = getattr(alphas, name)
        fields = list(spec.required_fields) + list(spec.industry_levels)
        result = pd.DataFrame(np.nan, index=exec_dates, columns=codes)
        t0 = time.time()
        for i, date in enumerate(exec_dates, 1):
            end_pos = dates.get_loc(date)
            start_pos = max(0, end_pos - args.lookback)
            window = {}
            for f in fields:
                if f in {"sector", "industry", "subindustry"}:
                    window[f] = industry[f].iloc[start_pos:end_pos + 1]
                elif f in {"open", "high", "low", "close", "vwap"}:
                    anchor = adj.iloc[end_pos].replace(0, np.nan)
                    window[f] = panels[f].iloc[start_pos:end_pos + 1].div(anchor, axis=1)
                else:
                    window[f] = panels[f].iloc[start_pos:end_pos + 1]
            fd = FactorData.from_mapping(window)
            with op.use_universe_mask(universe.iloc[start_pos:end_pos + 1]):
                res = fn(fd)
            result.loc[date] = res.iloc[-1]
            if i % 100 == 0:
                print(f"{name} {i}/{len(exec_dates)} elapsed={time.time()-t0:.0f}s", flush=True)
        result = result.where(universe.loc[exec_dates])
        result.to_parquet(out / f"{name}.parquet")
        summary.append({
            "factor": name,
            "calculation_method": "strict_daily_signal_anchor",
            "is_signal_anchor_exact": True,
            "valid_values": int(result.notna().sum().sum()),
            "elapsed_seconds": time.time() - t0,
        })
        print(f"STRICT {name} done {time.time()-t0:.0f}s valid={summary[-1]['valid_values']}", flush=True)
        (out / "strict_run_summary.csv").write_text(pd.DataFrame(summary).to_csv(index=False))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print("\nDONE")


if __name__ == "__main__":
    main()
