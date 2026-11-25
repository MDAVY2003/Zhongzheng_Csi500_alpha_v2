from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import factor_analysis as fa  # noqa: E402
from signal_decay import (  # noqa: E402
    calculate_delayed_peak_effective_window,
    calculate_rankic_decay_horizon,
    classify_decay_curve,
    summarize_delayed_peak,
    summarize_factor_lifetime,
    summarize_normal_decay,
)


class SignalAlignedOpenReturnTest(unittest.TestCase):
    def test_d1_and_d2_follow_global_calendar_positions(self) -> None:
        dates = pd.date_range("2022-12-28", periods=5, freq="B")
        opens = pd.DataFrame({"A": [10.0, 11.0, 22.0, 44.0, 88.0]}, index=dates)

        d1 = fa.calculate_signal_aligned_open_returns(opens, horizon=1)
        d2 = fa.calculate_signal_aligned_open_returns(opens, horizon=2)

        self.assertAlmostEqual(d1.loc[dates[0], "A"], 22.0 / 11.0 - 1.0)
        self.assertAlmostEqual(d2.loc[dates[0], "A"], 44.0 / 22.0 - 1.0)
        self.assertTrue(pd.isna(d1.loc[dates[-2], "A"]))

    def test_missing_open_does_not_skip_to_later_quote(self) -> None:
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        opens = pd.DataFrame(
            {
                "A": [10.0, 10.0, 11.0, 12.0, 13.0],
                "B": [20.0, np.nan, 22.0, 23.0, 24.0],
            },
            index=dates,
        )

        result = fa.calculate_signal_aligned_open_returns(opens, horizon=1)

        self.assertTrue(pd.isna(result.loc[dates[0], "B"]))
        self.assertAlmostEqual(result.loc[dates[0], "A"], 0.1)


class RankIcDecayTest(unittest.TestCase):
    def test_peak_effective_window_is_contiguous_and_stops_before_rebound(self) -> None:
        decay = pd.DataFrame(
            {
                "horizon": range(1, 11),
                "mean_rank_ic": [
                    0.010, 0.019, 0.021, 0.035, 0.040,
                    0.025, 0.018, 0.030, 0.010, 0.005,
                ],
            }
        )

        result = calculate_delayed_peak_effective_window(
            decay,
            factor_name="alpha_test",
            peak_horizon=5,
            peak_rankic=0.040,
        )

        self.assertEqual(result["effective_start_horizon"], 3)
        self.assertEqual(result["effective_end_horizon"], 6)
        self.assertEqual(result["effective_window_length"], 4)
        self.assertEqual(result["effective_horizons"], "D3,D4,D5,D6")
        self.assertFalse(result["left_censored"])
        self.assertFalse(result["right_censored"])

    def test_peak_effective_window_marks_both_observation_boundaries_censored(self) -> None:
        decay = pd.DataFrame(
            {
                "horizon": range(1, 11),
                "mean_rank_ic": [0.021, 0.022, 0.025, 0.030, 0.040] + [0.025] * 5,
            }
        )

        result = calculate_delayed_peak_effective_window(
            decay,
            factor_name="alpha_test",
            peak_horizon=5,
            peak_rankic=0.040,
        )

        self.assertEqual(result["effective_start_horizon"], 1)
        self.assertEqual(result["effective_end_horizon"], 10)
        self.assertTrue(result["left_censored"])
        self.assertTrue(result["right_censored"])

    def test_peak_rankic_must_match_decay_summary(self) -> None:
        decay = pd.DataFrame(
            {"horizon": range(1, 11), "mean_rank_ic": [0.01] * 4 + [0.04] + [0.01] * 5}
        )

        with self.assertRaisesRegex(ValueError, "peak_rankic与decay汇总不一致"):
            calculate_delayed_peak_effective_window(
                decay,
                factor_name="alpha_test",
                peak_horizon=5,
                peak_rankic=0.039,
            )

    def test_horizons_keep_the_same_signal_row(self) -> None:
        dates = pd.date_range("2020-01-01", periods=4, freq="B")
        codes = list("ABCDE")
        factor = pd.DataFrame([[1, 2, 3, 4, 5]], index=dates[:1], columns=codes)
        opens = pd.DataFrame(
            [
                [100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100],
                [101, 102, 103, 104, 105],
                [106, 105, 104, 103, 102],
            ],
            index=dates,
            columns=codes,
            dtype=float,
        )
        universe = pd.DataFrame(True, index=dates[:1], columns=codes)

        _, _, d1 = calculate_rankic_decay_horizon(
            factor,
            opens,
            universe,
            factor_name="alpha_test",
            direction=1,
            horizon=1,
            min_obs=5,
        )
        _, _, d2 = calculate_rankic_decay_horizon(
            factor,
            opens,
            universe,
            factor_name="alpha_test",
            direction=1,
            horizon=2,
            min_obs=5,
        )

        self.assertAlmostEqual(d1["mean_rank_ic"], 1.0)
        self.assertAlmostEqual(d2["mean_rank_ic"], -1.0)

    def test_lifetime_stops_at_first_drop_even_if_curve_rebounds(self) -> None:
        decay = pd.DataFrame(
            {
                "horizon": [1, 2, 3, 4],
                "mean_rank_ic": [0.040, 0.030, 0.018, 0.025],
            }
        )

        result = summarize_factor_lifetime(
            decay,
            factor_name="alpha_test",
            direction=1,
        )

        self.assertEqual(result["lifetime"], 2)
        self.assertEqual(result["first_below_horizon"], 3)
        self.assertFalse(result["censored"])

    def test_lifetime_is_censored_when_threshold_never_breaks(self) -> None:
        decay = pd.DataFrame(
            {
                "horizon": [1, 2, 3],
                "mean_rank_ic": [0.040, 0.030, 0.021],
            }
        )

        result = summarize_factor_lifetime(
            decay,
            factor_name="alpha_test",
            direction=-1,
        )

        self.assertTrue(result["censored"])
        self.assertTrue(pd.isna(result["lifetime"]))
        self.assertEqual(result["lifetime_lower_bound"], 3)

    def test_nonpositive_d1_is_invalid(self) -> None:
        decay = pd.DataFrame(
            {"horizon": [1, 2], "mean_rank_ic": [0.0, 0.01]}
        )

        result = summarize_factor_lifetime(
            decay,
            factor_name="alpha_test",
            direction=1,
        )

        self.assertEqual(result["lifetime_status"], "invalid_d1")
        self.assertFalse(result["censored"])

    def test_float_tolerance_keeps_curve_normal(self) -> None:
        decay = pd.DataFrame(
            {
                "horizon": range(1, 11),
                "mean_rank_ic": [0.04, 0.04 + 5e-13] + [0.03] * 8,
            }
        )

        result = classify_decay_curve(
            decay,
            factor_name="alpha_test",
            direction=1,
        )

        self.assertEqual(result["decay_type"], "normal")

    def test_normal_lifetime_stops_at_first_half_threshold_break(self) -> None:
        decay = pd.DataFrame(
            {
                "horizon": range(1, 11),
                "mean_rank_ic": [0.04, 0.03, 0.018, 0.025] + [0.01] * 6,
            }
        )
        classified = classify_decay_curve(
            decay,
            factor_name="alpha_test",
            direction=1,
        )
        result = summarize_normal_decay(classified)

        self.assertEqual(result["lifetime"], 2)
        self.assertFalse(result["censored"])

    def test_delayed_peak_uses_peak_based_post_lifetime(self) -> None:
        decay = pd.DataFrame(
            {
                "horizon": range(1, 11),
                "mean_rank_ic": [
                    0.015,
                    0.025,
                    0.040,
                    0.030,
                    0.018,
                    0.015,
                    0.01,
                    0.0,
                    -0.01,
                    -0.02,
                ],
            }
        )
        classified = classify_decay_curve(
            decay,
            factor_name="alpha_test",
            direction=1,
        )
        result = summarize_delayed_peak(classified)

        self.assertEqual(result["decay_type"], "delayed_peak")
        self.assertEqual(result["peak_horizon"], 3)
        self.assertAlmostEqual(result["peak_rankic"], 0.04)
        self.assertEqual(result["post_peak_lifetime"], 2)

    def test_peak_at_d10_is_right_censored(self) -> None:
        decay = pd.DataFrame(
            {
                "horizon": range(1, 11),
                "mean_rank_ic": np.linspace(0.01, 0.10, 10),
            }
        )
        classified = classify_decay_curve(
            decay,
            factor_name="alpha_test",
            direction=1,
        )
        result = summarize_delayed_peak(classified)

        self.assertEqual(result["peak_horizon"], 10)
        self.assertTrue(result["censored"])
        self.assertTrue(pd.isna(result["post_peak_lifetime"]))
        self.assertEqual(result["post_peak_lifetime_lower_bound"], 0)


if __name__ == "__main__":
    unittest.main()
