import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from exact_signal_anchor import (  # noqa: E402
    calculate_exact_alpha004,
    calculate_exact_alpha067,
    calculate_exact_alpha088,
)
from factor_data import FactorData  # noqa: E402


class ExactSignalAnchorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.date_range("2020-01-01", periods=110, freq="D")
        self.codes = pd.Index(["A", "B", "C"])
        self.mask = pd.DataFrame(True, index=self.dates, columns=self.codes)
        self.adjustment = pd.DataFrame(
            1.0, index=self.dates, columns=self.codes
        )
        self.adjustment.loc[self.dates[9]:, "A"] = 2.0

    def _panel(self, values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=self.dates, columns=self.codes)

    def test_alpha004_reanchors_all_historical_cross_sections(self) -> None:
        low = self._panel(
            np.column_stack(
                [
                    np.linspace(1, 20, len(self.dates)),
                    np.linspace(5, 6, len(self.dates)),
                    np.linspace(9, 7, len(self.dates)),
                ]
            )
        )
        result = calculate_exact_alpha004(
            FactorData(low=low), self.adjustment, self.mask
        )
        end = 9
        anchored = low.iloc[1:10].div(self.adjustment.iloc[end], axis=1)
        ranked = anchored.rank(axis=1, pct=True)
        expected = -ranked.rolling(9, min_periods=9).apply(
            lambda values: float(
                (values < values[-1]).sum()
                + ((values == values[-1]).sum() + 1) / 2
            )
            / len(values),
            raw=True,
        ).iloc[-1]
        pd.testing.assert_series_equal(
            result.iloc[end], expected, check_names=False
        )

    def test_alpha067_returns_aligned_panel(self) -> None:
        t = np.arange(len(self.dates), dtype=float)[:, None]
        scale = np.array([[1.0, 1.3, 1.7]])
        high = self._panel(10 + t * scale)
        vwap = self._panel(8 + np.sin(t / 5) * scale + t * 0.01)
        volume = self._panel(100 + (t % 17) * scale + t * scale)
        groups = pd.DataFrame(
            [["S1", "S1", "S2"]] * len(self.dates),
            index=self.dates,
            columns=self.codes,
        )
        result = calculate_exact_alpha067(
            FactorData(
                high=high,
                vwap=vwap,
                volume=volume,
                amount=volume * vwap,
                sector=groups,
                subindustry=groups,
            ),
            self.adjustment,
            self.mask,
        )
        self.assertTrue(result.index.equals(self.dates))
        self.assertTrue(result.columns.equals(self.codes))
        self.assertTrue(result.iloc[:24].isna().all().all())

    def test_alpha088_preserves_second_branch_missing_values(self) -> None:
        t = np.arange(len(self.dates), dtype=float)[:, None]
        scale = np.array([[1.0, 1.4, 2.0]])
        common = self._panel(20 + t * scale)
        volume = self._panel(100 + t * scale)
        result = calculate_exact_alpha088(
            FactorData(
                open=common,
                high=common + 2,
                low=common - 2,
                close=common + np.sin(t / 7) * scale,
                volume=volume,
                amount=volume * common,
            ),
            self.adjustment,
            self.mask,
        )
        self.assertTrue(result.index.equals(self.dates))
        self.assertTrue(result.columns.equals(self.codes))
        self.assertTrue(result.iloc[:7].isna().all().all())
        self.assertFalse(np.isinf(result.to_numpy()).any())


if __name__ == "__main__":
    unittest.main()
