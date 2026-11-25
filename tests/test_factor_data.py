from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import factor_data  # noqa: E402


class FactorDataTest(unittest.TestCase):
    def test_industry_intervals_expand_without_future_backfill(self) -> None:
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        membership = pd.DataFrame(
            [
                {
                    "code": "AAA",
                    "sector_code": "S1",
                    "industry_code": "I1",
                    "subindustry_code": "SI1",
                    "in_date": "2025-01-01",
                    "out_date": "2025-01-02",
                },
                {
                    "code": "AAA",
                    "sector_code": "S2",
                    "industry_code": "I2",
                    "subindustry_code": "SI2",
                    "in_date": "2025-01-03",
                    "out_date": None,
                },
                {
                    "code": "BBB",
                    "sector_code": "S1",
                    "industry_code": "I3",
                    "subindustry_code": "SI3",
                    "in_date": "2025-01-02",
                    "out_date": None,
                },
            ]
        )

        panels = factor_data.build_sw_industry_panels(
            membership,
            dates=dates,
            codes=["AAA", "BBB"],
        )

        self.assertEqual(panels["sector"].loc[dates[0], "AAA"], "S1")
        self.assertEqual(panels["sector"].loc[dates[2], "AAA"], "S2")
        self.assertTrue(pd.isna(panels["sector"].loc[dates[0], "BBB"]))
        self.assertEqual(panels["industry"].loc[dates[1], "BBB"], "I3")
        self.assertEqual(
            panels["subindustry"].loc[dates[-1], "AAA"],
            "SI2",
        )


if __name__ == "__main__":
    unittest.main()
