from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import alphas  # noqa: E402


class SimpleAlphasTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.date_range("2025-01-02", periods=3, freq="D")
        self.columns = ["AAA", "BBB", "CCC"]

        self.open = pd.DataFrame(
            [[9.0, 20.0, 24.0], [11.0, 18.0, 30.0], [10.0, 21.0, 27.0]],
            index=self.dates,
            columns=self.columns,
        )
        self.close = pd.DataFrame(
            [[10.0, 20.0, 20.0], [12.0, 20.0, 30.0], [11.0, 19.0, 30.0]],
            index=self.dates,
            columns=self.columns,
        )
        self.high = pd.DataFrame(
            [[11.0, 22.0, 31.0], [13.0, 21.0, 32.0], [12.0, 22.0, 31.0]],
            index=self.dates,
            columns=self.columns,
        )
        self.low = pd.DataFrame(
            [[8.0, 18.0, 19.0], [10.0, 17.0, 28.0], [9.0, 18.0, 26.0]],
            index=self.dates,
            columns=self.columns,
        )
        self.vwap = pd.DataFrame(
            [[9.0, 22.0, 30.0], [12.0, 19.0, 31.0], [10.5, 20.0, 29.0]],
            index=self.dates,
            columns=self.columns,
        )
        self.volume = pd.DataFrame(
            [[100.0, 300.0, 500.0], [120.0, 250.0, 500.0], [110.0, 280.0, 550.0]],
            index=self.dates,
            columns=self.columns,
        )

    def test_alpha_012_matches_original_formula(self) -> None:
        result = alphas.alpha_012(self.volume, self.close)
        expected = np.sign(self.volume.diff(1)) * (-1 * self.close.diff(1))
        pd.testing.assert_frame_equal(result, expected)
        self.assertTrue(result.iloc[0].isna().all())

    def test_alpha_033_is_cross_sectional_rank(self) -> None:
        result = alphas.alpha_033(self.open, self.close)
        expected_first_day = pd.Series(
            {"AAA": 1 / 3, "BBB": 2 / 3, "CCC": 1.0},
            name=self.dates[0],
        )
        pd.testing.assert_series_equal(result.iloc[0], expected_first_day)

    def test_alpha_041_matches_original_formula(self) -> None:
        result = alphas.alpha_041(self.high, self.low, self.vwap)
        expected = np.sqrt(self.high * self.low) - self.vwap
        pd.testing.assert_frame_equal(result, expected)

    def test_alpha_042_uses_two_independent_ranks(self) -> None:
        close = self.close.copy()
        close.loc[self.dates[0], "CCC"] = 30.0
        result = alphas.alpha_042(self.vwap, close)
        expected_first_day = pd.Series(
            {"AAA": 1.0, "BBB": 1.5, "CCC": 2 / 3},
            name=self.dates[0],
        )
        pd.testing.assert_series_equal(result.iloc[0], expected_first_day)

    def test_alpha_101_keeps_point_zero_zero_one_constant(self) -> None:
        result = alphas.alpha_101(
            self.close,
            self.open,
            self.high,
            self.low,
        )
        expected = (self.close - self.open) / (
            (self.high - self.low) + 0.001
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_inputs_must_have_identical_columns(self) -> None:
        misaligned = self.close.rename(columns={"CCC": "DDD"})
        with self.assertRaisesRegex(ValueError, "股票列不一致"):
            alphas.alpha_033(self.open, misaligned)


if __name__ == "__main__":
    unittest.main()
