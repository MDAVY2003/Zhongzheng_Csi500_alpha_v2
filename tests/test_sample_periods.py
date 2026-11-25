from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_in_sample_factor_analysis import prepare_in_sample_returns  # noqa: E402
from sample_periods import (  # noqa: E402
    IN_SAMPLE_PERIOD,
    OUT_OF_SAMPLE_PERIOD,
    WARMUP_PERIOD,
)


class SamplePeriodsTest(unittest.TestCase):
    def test_fixed_period_boundaries(self) -> None:
        self.assertEqual(WARMUP_PERIOD.start, pd.Timestamp("2017-01-01"))
        self.assertEqual(WARMUP_PERIOD.end, pd.Timestamp("2017-12-31"))
        self.assertEqual(IN_SAMPLE_PERIOD.start, pd.Timestamp("2018-01-01"))
        self.assertEqual(IN_SAMPLE_PERIOD.end, pd.Timestamp("2022-12-31"))
        self.assertEqual(OUT_OF_SAMPLE_PERIOD.start, pd.Timestamp("2023-01-01"))
        self.assertEqual(OUT_OF_SAMPLE_PERIOD.end, pd.Timestamp("2025-12-31"))

    def test_period_selection_excludes_warmup_and_out_of_sample(self) -> None:
        dates = pd.to_datetime(
            ["2017-12-29", "2018-01-02", "2022-12-30", "2023-01-03"]
        )
        frame = pd.DataFrame({"A": [90.0, 100.0, 110.0, 220.0]}, index=dates)
        selected = IN_SAMPLE_PERIOD.select(frame)
        self.assertEqual(
            selected.index.tolist(),
            [pd.Timestamp("2018-01-02"), pd.Timestamp("2022-12-30")],
        )

    def test_forward_return_does_not_cross_into_out_of_sample(self) -> None:
        dates = pd.to_datetime(
            [
                "2017-12-29",
                "2022-12-29",
                "2022-12-30",
                "2023-01-03",
            ]
        )
        close = pd.DataFrame({"A": [90.0, 100.0, 110.0, 220.0]}, index=dates)
        result = prepare_in_sample_returns(close)
        self.assertAlmostEqual(result.loc[pd.Timestamp("2022-12-29"), "A"], 0.10)
        self.assertTrue(result.loc[pd.Timestamp("2022-12-30")].isna().all())
        self.assertNotIn(pd.Timestamp("2023-01-03"), result.index)


if __name__ == "__main__":
    unittest.main()
