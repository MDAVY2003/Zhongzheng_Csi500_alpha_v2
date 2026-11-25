from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import operators as op  # noqa: E402


class OperatorsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.date_range("2025-01-02", periods=4, freq="D")
        self.panel = pd.DataFrame(
            {
                "AAA": [3.0, 1.0, 2.0, 4.0],
                "BBB": [1.0, 2.0, 3.0, 1.0],
                "CCC": [2.0, 3.0, 1.0, 2.0],
            },
            index=self.dates,
        )

    def test_rank_is_cross_sectional(self) -> None:
        result = op.rank(self.panel)
        expected_first_day = pd.Series(
            {"AAA": 1.0, "BBB": 1 / 3, "CCC": 2 / 3},
            name=self.dates[0],
        )
        pd.testing.assert_series_equal(result.iloc[0], expected_first_day)

    def test_cross_sectional_operators_respect_dynamic_universe(self) -> None:
        mask = pd.DataFrame(True, index=self.dates, columns=self.panel.columns)
        mask.loc[self.dates[0], "AAA"] = False
        mask.loc[self.dates[1], "CCC"] = False

        with op.use_universe_mask(mask):
            ranked = op.rank(self.panel)
            scaled = op.scale(self.panel)

        self.assertTrue(pd.isna(ranked.loc[self.dates[0], "AAA"]))
        self.assertEqual(ranked.loc[self.dates[0], "BBB"], 0.5)
        self.assertEqual(ranked.loc[self.dates[0], "CCC"], 1.0)
        self.assertTrue(pd.isna(scaled.loc[self.dates[1], "CCC"]))
        self.assertAlmostEqual(scaled.loc[self.dates[1]].abs().sum(), 1.0)

    def test_universe_context_does_not_mask_time_series_inputs(self) -> None:
        mask = pd.DataFrame(True, index=self.dates, columns=self.panel.columns)
        mask.loc[self.dates[1], "AAA"] = False
        with op.use_universe_mask(mask):
            rolling = op.rolling_mean(self.panel, 2)
        self.assertEqual(rolling.loc[self.dates[1], "AAA"], 2.0)

    def test_ts_rank_is_per_stock_over_time(self) -> None:
        result = op.ts_rank(self.panel, window=3)
        self.assertTrue(result.iloc[:2].isna().all().all())
        self.assertEqual(result.loc[self.dates[2], "AAA"], 2 / 3)
        self.assertEqual(result.loc[self.dates[2], "BBB"], 1.0)
        self.assertEqual(result.loc[self.dates[2], "CCC"], 1 / 3)

    def test_delay_and_delta_follow_date_axis(self) -> None:
        delayed = op.delay(self.panel, period=1)
        changed = op.delta(self.panel, period=1)
        self.assertEqual(delayed.loc[self.dates[1], "AAA"], 3.0)
        self.assertEqual(changed.loc[self.dates[1], "AAA"], -2.0)

    def test_rolling_statistics(self) -> None:
        pd.testing.assert_frame_equal(
            op.rolling_mean(self.panel, 2),
            self.panel.rolling(2).mean(),
        )
        pd.testing.assert_frame_equal(
            op.rolling_sum(self.panel, 2),
            self.panel.rolling(2).sum(),
        )
        pd.testing.assert_frame_equal(
            op.rolling_std(self.panel, 2),
            self.panel.rolling(2).std(),
        )
        pd.testing.assert_frame_equal(
            op.ts_min(self.panel, 2),
            self.panel.rolling(2).min(),
        )
        pd.testing.assert_frame_equal(
            op.ts_max(self.panel, 2),
            self.panel.rolling(2).max(),
        )

    def test_rolling_correlation_and_covariance(self) -> None:
        right = self.panel * 2.0
        correlation = op.rolling_correlation(self.panel, right, 3)
        covariance = op.rolling_covariance(self.panel, right, 3)
        pd.testing.assert_frame_equal(
            correlation,
            self.panel.rolling(3).corr(right),
        )
        pd.testing.assert_frame_equal(
            covariance,
            self.panel.rolling(3).cov(right),
        )

    def test_zero_variance_correlation_never_returns_infinity(self) -> None:
        dates = pd.date_range("2025-01-01", periods=4, freq="D")
        left = pd.DataFrame({"AAA": [1.0, 1.0, 1.0, 1.0]}, index=dates)
        right = pd.DataFrame({"AAA": [1.0, 2.0, 3.0, 4.0]}, index=dates)
        result = op.rolling_correlation(left, right, 3)
        self.assertFalse(np.isinf(result.to_numpy()).any())

    def test_ts_arg_positions_start_from_oldest_observation(self) -> None:
        maximum_position = op.ts_argmax(self.panel[["AAA"]], 3)
        minimum_position = op.ts_argmin(self.panel[["AAA"]], 3)
        self.assertEqual(maximum_position.loc[self.dates[2], "AAA"], 1.0)
        self.assertEqual(minimum_position.loc[self.dates[2], "AAA"], 2.0)
        self.assertEqual(maximum_position.loc[self.dates[3], "AAA"], 3.0)

    def test_scale_is_cross_sectional(self) -> None:
        result = op.scale(self.panel, k=2.0)
        absolute_sums = result.abs().sum(axis=1)
        expected = pd.Series(2.0, index=self.dates)
        pd.testing.assert_series_equal(absolute_sums, expected)

    def test_sign_and_abs(self) -> None:
        values = pd.Series(
            [-2.0, 0.0, 3.0, np.nan],
            index=self.dates,
        )
        pd.testing.assert_series_equal(
            op.sign(values),
            pd.Series([-1.0, 0.0, 1.0, np.nan], index=self.dates),
        )
        pd.testing.assert_series_equal(
            op.abs(values),
            pd.Series([2.0, 0.0, 3.0, np.nan], index=self.dates),
        )

    def test_time_operators_reject_descending_dates(self) -> None:
        descending = self.panel.sort_index(ascending=False)
        with self.assertRaisesRegex(ValueError, "升序"):
            op.ts_rank(descending, 2)

    def test_rank_requires_wide_dataframe(self) -> None:
        with self.assertRaisesRegex(TypeError, "横截面"):
            op.rank(self.panel["AAA"])

    def test_decay_linear_weights_recent_values_more(self) -> None:
        result = op.decay_linear(self.panel[["AAA"]], 3)
        expected = (3.0 * 1 + 1.0 * 2 + 2.0 * 3) / 6
        self.assertAlmostEqual(result.loc[self.dates[2], "AAA"], expected)
        self.assertTrue(result.iloc[:2].isna().all().all())

    def test_elementwise_min_and_max(self) -> None:
        right = self.panel.iloc[::-1].set_axis(self.dates)
        pd.testing.assert_frame_equal(
            op.elementwise_min(self.panel, right),
            pd.DataFrame(
                np.minimum(self.panel, right),
                index=self.dates,
                columns=self.panel.columns,
            ),
        )
        pd.testing.assert_frame_equal(
            op.elementwise_max(self.panel, 2.0),
            pd.DataFrame(
                np.maximum(self.panel, 2.0),
                index=self.dates,
                columns=self.panel.columns,
            ),
        )

    def test_industry_neutralize_removes_group_means(self) -> None:
        groups = pd.Series(
            {"AAA": "industry_1", "BBB": "industry_1", "CCC": "industry_2"}
        )
        result = op.industry_neutralize(self.panel, groups)
        expected_first_day = pd.Series(
            {"AAA": 1.0, "BBB": -1.0, "CCC": 0.0},
            name=self.dates[0],
        )
        pd.testing.assert_series_equal(result.iloc[0], expected_first_day)
        group_one_mean = result[["AAA", "BBB"]].mean(axis=1)
        np.testing.assert_allclose(group_one_mean, 0.0)

    def test_log_rolling_product_and_signed_power(self) -> None:
        values = pd.Series(
            [1.0, np.e, 0.0, -1.0],
            index=self.dates,
        )
        logged = op.log(values)
        self.assertAlmostEqual(logged.iloc[0], 0.0)
        self.assertAlmostEqual(logged.iloc[1], 1.0)
        self.assertTrue(logged.iloc[2:].isna().all())

        product = op.rolling_product(self.panel[["AAA"]], 3)
        self.assertEqual(product.loc[self.dates[2], "AAA"], 6.0)

        base = pd.Series([-2.0, 3.0, -4.0, 5.0], index=self.dates)
        exponent = pd.Series([2.0, 2.0, 0.5, 1.0], index=self.dates)
        expected = pd.Series([-4.0, 9.0, -2.0, 5.0], index=self.dates)
        pd.testing.assert_series_equal(
            op.signed_power(base, exponent),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
