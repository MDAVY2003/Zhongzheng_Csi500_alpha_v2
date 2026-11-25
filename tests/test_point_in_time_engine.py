"""通用点时因子引擎测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import alphas  # noqa: E402
from factor_data import FactorData  # noqa: E402
from point_in_time_engine import calculate_point_in_time_factors  # noqa: E402


class PointInTimeEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(20260806)
        dates = pd.bdate_range("2020-01-01", periods=80)
        codes = pd.Index([f"S{i}" for i in range(8)])
        close = pd.DataFrame(
            30 + rng.normal(0, 0.5, (len(dates), len(codes))).cumsum(axis=0),
            index=dates,
            columns=codes,
        )
        open_price = close * (1 + rng.normal(0, 0.005, close.shape))
        high = np.maximum(open_price, close) * 1.01
        low = np.minimum(open_price, close) * 0.99
        volume = pd.DataFrame(
            rng.lognormal(13, 0.4, close.shape), index=dates, columns=codes
        )
        self.data = FactorData(
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            vwap=(high + low + close) / 3,
            returns=close.pct_change(fill_method=None),
        )
        self.adj = pd.DataFrame(1.0, index=dates, columns=codes)
        self.adj.loc[dates[25]:, codes[:3]] = 1.5
        self.adj.loc[dates[55]:, codes[3:5]] = 0.8

    def test_exact_vectorized_formulas_match_signal_date_anchor(self) -> None:
        names = ["alpha006", "alpha012", "alpha041", "alpha101"]
        actual = calculate_point_in_time_factors(
            self.data,
            self.adj,
            factor_names=names,
        )
        for date in self.data.index:
            anchor = self.adj.loc[date]
            anchored = FactorData.from_mapping(
                {
                    name: (
                        panel.div(anchor, axis=1)
                        if name in {"open", "high", "low", "close", "vwap"}
                        else panel
                    )
                    for name, panel in self.data.as_dict().items()
                }
            )
            for name in names:
                expected = getattr(alphas, name)(anchored).loc[date]
                pd.testing.assert_series_equal(
                    actual[name].loc[date],
                    expected,
                    check_names=False,
                    rtol=1e-10,
                    atol=1e-12,
                )

    def test_matrix_exponent_signed_power_formula_runs(self) -> None:
        result = calculate_point_in_time_factors(
            self.data,
            self.adj,
            factor_names=["alpha084"],
        )["alpha084"]
        self.assertEqual(result.shape, self.data.close.shape)
        self.assertFalse(np.isinf(result.to_numpy(dtype=float)).any())


if __name__ == "__main__":
    unittest.main()
