from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import factor_analysis as fa  # noqa: E402


class FactorAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.date_range("2025-01-02", periods=6, freq="D")
        self.codes = ["A", "B", "C", "D", "E", "F"]
        values = np.tile(np.arange(1.0, 7.0), (6, 1))
        self.factor = pd.DataFrame(
            values,
            index=self.dates,
            columns=self.codes,
        )
        self.positive_returns = self.factor * 0.01
        self.negative_returns = -self.factor * 0.01

    def test_perfect_positive_rank_ic(self) -> None:
        rank_ic = fa.calculate_daily_rank_ic(
            self.factor,
            self.positive_returns,
            min_obs=6,
        )
        np.testing.assert_allclose(rank_ic, 1.0)

    def test_perfect_negative_rank_ic(self) -> None:
        rank_ic = fa.calculate_daily_rank_ic(
            self.factor,
            self.negative_returns,
            min_obs=6,
        )
        np.testing.assert_allclose(rank_ic, -1.0)

    def test_rank_ic_can_be_one_when_pearson_ic_is_not(self) -> None:
        nonlinear_returns = self.factor**2
        ic = fa.calculate_daily_ic(
            self.factor,
            nonlinear_returns,
            min_obs=6,
        )
        rank_ic = fa.calculate_daily_rank_ic(
            self.factor,
            nonlinear_returns,
            min_obs=6,
        )
        self.assertTrue((ic < 1.0).all())
        np.testing.assert_allclose(rank_ic, 1.0)

    def test_pearson_ic_is_stable_for_very_large_finite_values(self) -> None:
        large_factor = self.factor * 1e290
        ic = fa.calculate_daily_ic(
            large_factor,
            self.positive_returns,
            min_obs=6,
        )
        np.testing.assert_allclose(ic, 1.0)

    def test_missing_values_use_only_jointly_valid_stocks(self) -> None:
        factor = self.factor.copy()
        returns = self.positive_returns.copy()
        factor.loc[self.dates[0], "C"] = np.nan
        returns.loc[self.dates[0], "B"] = np.nan

        coverage = fa.calculate_factor_coverage(factor, returns)
        self.assertEqual(coverage.loc[self.dates[0], "valid_count"], 4)
        self.assertAlmostEqual(
            coverage.loc[self.dates[0], "coverage"],
            4 / 6,
        )
        self.assertAlmostEqual(
            coverage.loc[self.dates[0], "missing_rate"],
            1 / 6,
        )

        ic = fa.calculate_daily_ic(factor, returns, min_obs=4)
        self.assertFalse(pd.isna(ic.loc[self.dates[0]]))

    def test_dynamic_universe_controls_coverage_and_cross_section(self) -> None:
        universe = pd.DataFrame(
            False,
            index=self.dates,
            columns=self.codes,
        )
        universe.loc[:, ["A", "B", "C", "D"]] = True
        returns = self.positive_returns.copy()
        returns.loc[:, ["E", "F"]] = [[100.0, -100.0]] * len(self.dates)

        coverage = fa.calculate_factor_coverage(
            self.factor,
            returns,
            universe_mask=universe,
        )
        self.assertTrue(coverage["universe_count"].eq(4).all())
        self.assertTrue(coverage["valid_count"].eq(4).all())
        self.assertTrue(coverage["coverage"].eq(1.0).all())

        analysis = fa.analyze_single_factor(
            self.factor,
            returns,
            factor_name="dynamic_universe",
            min_obs=4,
            n_groups=2,
            universe_mask=universe,
        )
        np.testing.assert_allclose(analysis["daily_rank_ic"], 1.0)

    def test_minimum_observation_rule_returns_nan(self) -> None:
        factor = self.factor.copy()
        returns = self.positive_returns.copy()
        factor.loc[self.dates[0], ["C", "D", "E", "F"]] = np.nan

        ic = fa.calculate_daily_ic(factor, returns, min_obs=3)
        rank_ic = fa.calculate_daily_rank_ic(
            factor,
            returns,
            min_obs=3,
        )
        self.assertTrue(pd.isna(ic.loc[self.dates[0]]))
        self.assertTrue(pd.isna(rank_ic.loc[self.dates[0]]))

    def test_misaligned_inputs_raise_clear_errors(self) -> None:
        wrong_index = self.positive_returns.copy()
        wrong_index.index = wrong_index.index + pd.Timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "index不一致"):
            fa.calculate_daily_ic(self.factor, wrong_index, min_obs=2)

        wrong_columns = self.positive_returns.rename(columns={"F": "X"})
        with self.assertRaisesRegex(ValueError, "columns不一致"):
            fa.calculate_daily_ic(self.factor, wrong_columns, min_obs=2)

    def test_infinite_input_is_rejected(self) -> None:
        factor = self.factor.copy()
        factor.loc[self.dates[0], "A"] = np.inf
        with self.assertRaisesRegex(ValueError, "正负无穷"):
            fa.calculate_daily_ic(
                factor,
                self.positive_returns,
                min_obs=2,
            )

    def test_group_returns_increase_with_factor(self) -> None:
        groups = fa.calculate_group_returns(
            self.factor,
            self.positive_returns,
            min_obs=6,
            n_groups=3,
        )
        self.assertTrue((groups["group_3"] > groups["group_1"]).all())
        long_short = groups["group_3"] - groups["group_1"]
        self.assertTrue((long_short > 0).all())

    def test_duplicate_factor_values_do_not_break_grouping(self) -> None:
        duplicate_factor = self.factor.copy()
        duplicate_factor.loc[:, :] = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
        groups = fa.calculate_group_returns(
            duplicate_factor,
            self.positive_returns,
            min_obs=6,
            n_groups=3,
        )
        self.assertEqual(
            groups.columns.tolist(),
            ["group_1", "group_2", "group_3"],
        )
        self.assertTrue(groups.notna().all().all())

    def test_vectorized_groups_match_qcut_boundaries(self) -> None:
        dates = pd.DatetimeIndex(["2025-01-02"])
        codes = list("ABCDEFG")
        factor = pd.DataFrame(
            [np.arange(1.0, 8.0)],
            index=dates,
            columns=codes,
        )
        returns = factor / 100.0
        groups = fa.calculate_group_returns(
            factor,
            returns,
            min_obs=7,
            n_groups=3,
        )
        labels = pd.qcut(
            factor.iloc[0].rank(method="first"),
            q=3,
            labels=False,
        ) + 1
        expected = returns.iloc[0].groupby(labels).mean()
        for group in range(1, 4):
            self.assertAlmostEqual(
                groups.loc[dates[0], f"group_{group}"],
                expected.loc[group],
            )

    def test_constant_cross_section_has_no_arbitrary_group_return(self) -> None:
        constant_factor = self.factor.copy()
        constant_factor.loc[:, :] = 1.0
        groups = fa.calculate_group_returns(
            constant_factor,
            self.positive_returns,
            min_obs=6,
            n_groups=3,
        )
        self.assertTrue(groups.isna().all().all())

    def test_icir_uses_sample_standard_deviation(self) -> None:
        ic = pd.Series(
            [0.1, -0.1, 0.2, 0.0],
            index=self.dates[:4],
        )
        rank_ic = pd.Series(0.2, index=self.dates[:4])
        coverage = pd.Series(0.8, index=self.dates[:4])
        valid_count = pd.Series(80, index=self.dates[:4])

        summary = fa.summarize_ic(
            ic,
            rank_ic,
            coverage,
            valid_count,
            factor_name="example",
        )
        self.assertAlmostEqual(summary["ic_mean"], ic.mean())
        self.assertAlmostEqual(summary["ic_std"], ic.std(ddof=1))
        self.assertAlmostEqual(
            summary["icir"],
            ic.mean() / ic.std(ddof=1),
        )
        self.assertEqual(summary["ic_win_rate"], 0.5)
        self.assertTrue(pd.isna(summary["rank_icir"]))

    def test_forward_return_alignment_uses_next_close(self) -> None:
        dates = pd.date_range("2025-01-02", periods=3, freq="D")
        close = pd.DataFrame(
            {
                "A": [100.0, 110.0, 121.0],
                "B": [50.0, 45.0, 54.0],
            },
            index=dates,
        )
        original = close.copy()

        result = fa.calculate_forward_returns(close)

        self.assertAlmostEqual(result.loc[dates[0], "A"], 0.1)
        self.assertAlmostEqual(result.loc[dates[1], "A"], 0.1)
        self.assertAlmostEqual(result.loc[dates[0], "B"], -0.1)
        self.assertAlmostEqual(result.loc[dates[1], "B"], 0.2)
        self.assertTrue(result.loc[dates[-1]].isna().all())
        pd.testing.assert_frame_equal(close, original)

    def test_single_and_batch_analysis_outputs(self) -> None:
        positive = fa.analyze_single_factor(
            self.factor,
            self.positive_returns,
            factor_name="positive",
            min_obs=6,
            n_groups=3,
        )
        required_keys = {
            "summary",
            "daily_ic",
            "daily_rank_ic",
            "adjusted_daily_ic",
            "adjusted_daily_rank_ic",
            "daily_coverage",
            "daily_valid_count",
            "yearly_summary",
            "group_returns",
            "adjusted_group_returns",
            "long_short_returns",
            "adjusted_long_short_returns",
            "long_short_cumulative_net_value",
            "adjusted_long_short_cumulative_net_value",
        }
        self.assertTrue(required_keys.issubset(positive))
        self.assertEqual(
            positive["summary"]["suggested_direction"],
            1.0,
        )
        self.assertIn(2025, positive["yearly_summary"].index)

        batch = fa.analyze_factors(
            {
                "positive": self.factor,
                "negative": -self.factor,
            },
            self.positive_returns,
            min_obs=6,
            n_groups=3,
        )
        self.assertEqual(batch["summary"].shape[0], 2)
        directions = batch["summary"].set_index("factor_name")[
            "suggested_direction"
        ]
        self.assertEqual(directions["positive"], 1.0)
        self.assertEqual(directions["negative"], -1.0)

    def test_long_short_statistics_and_cumulative_net_value(self) -> None:
        returns = pd.Series(
            [0.10, np.nan, -0.05, 0.02],
            index=self.dates[:4],
        )
        net_value = fa.calculate_cumulative_net_value(returns)
        self.assertAlmostEqual(net_value.iloc[0], 1.10)
        self.assertTrue(pd.isna(net_value.iloc[1]))
        self.assertAlmostEqual(net_value.iloc[-1], 1.10 * 0.95 * 1.02)

        statistics = fa.summarize_long_short_returns(
            returns,
            annualization_factor=252,
        )
        valid = returns.dropna()
        self.assertAlmostEqual(statistics["long_short_mean"], valid.mean())
        self.assertAlmostEqual(
            statistics["long_short_std"],
            valid.std(ddof=1),
        )
        self.assertAlmostEqual(
            statistics["long_short_annualized_volatility"],
            valid.std(ddof=1) * np.sqrt(252),
        )
        self.assertAlmostEqual(
            statistics["long_short_annualized_sharpe"],
            valid.mean() / valid.std(ddof=1) * np.sqrt(252),
        )
        self.assertEqual(statistics["long_short_valid_days"], 3)

    def test_insufficient_coverage_is_explicitly_marked(self) -> None:
        factor = self.factor.copy()
        factor.loc[:, ["C", "D", "E", "F"]] = np.nan
        analysis = fa.analyze_single_factor(
            factor,
            self.positive_returns,
            factor_name="sparse",
            min_obs=3,
            n_groups=2,
        )
        summary = analysis["summary"]
        self.assertEqual(summary["analysis_status"], "insufficient_coverage")
        self.assertEqual(summary["max_valid_count"], 2)
        self.assertEqual(summary["valid_days"], 0)
        self.assertTrue(analysis["daily_ic"].isna().all())
        self.assertTrue(analysis["long_short_returns"].isna().all())

    def test_yearly_summary_contains_long_short_risk_metrics(self) -> None:
        analysis = fa.analyze_single_factor(
            self.factor,
            self.positive_returns,
            factor_name="positive",
            min_obs=6,
            n_groups=3,
        )
        yearly = analysis["yearly_summary"]
        self.assertIn("long_short_annualized_volatility", yearly.columns)
        self.assertIn("long_short_annualized_sharpe", yearly.columns)
        self.assertIn("long_short_final_net_value", yearly.columns)
        self.assertGreater(
            analysis["summary"]["long_short_final_net_value"],
            1.0,
        )

    def test_negative_direction_adjusts_daily_yearly_and_group_results(self) -> None:
        analysis = fa.analyze_single_factor(
            -self.factor,
            self.positive_returns,
            factor_name="negative",
            min_obs=6,
            n_groups=3,
        )
        summary = analysis["summary"]

        self.assertEqual(summary["selected_direction"], -1.0)
        np.testing.assert_allclose(analysis["daily_rank_ic"], -1.0)
        np.testing.assert_allclose(analysis["adjusted_daily_rank_ic"], 1.0)
        self.assertAlmostEqual(summary["adjusted_rank_ic"], 1.0)
        self.assertAlmostEqual(summary["adjusted_rank_ic_win_rate"], 1.0)
        pd.testing.assert_series_equal(
            analysis["adjusted_long_short_returns"],
            -analysis["long_short_returns"].rename(
                "adjusted_long_short_return"
            ),
        )
        pd.testing.assert_series_equal(
            analysis["adjusted_group_returns"]["adjusted_group_1"],
            analysis["group_returns"]["group_3"].rename(
                "adjusted_group_1"
            ),
        )
        yearly = analysis["yearly_summary"].loc[2025]
        self.assertAlmostEqual(yearly["rank_ic_mean"], -1.0)
        self.assertAlmostEqual(yearly["adjusted_rank_ic"], 1.0)
        self.assertAlmostEqual(yearly["adjusted_rank_ic_win_rate"], 1.0)

    def test_fixed_selection_rules(self) -> None:
        base = pd.Series(
            {
                "analysis_status": "success",
                "selected_direction": 1.0,
                "valid_rank_ic_day_ratio": 0.90,
                "adjusted_rank_ic": 0.020,
                "adjusted_rank_icir": 0.25,
                "adjusted_rank_ic_win_rate": 0.53,
                "adjusted_positive_rank_ic_years": 3,
                "rank_ic_top_5pct_abs_share": 0.20,
                # 以下两项只是诊断，即使很差也不能单独否决。
                "adjusted_long_short_annualized_sharpe": -10.0,
                "adjusted_group_monotonicity_ratio": 0.0,
            }
        )
        self.assertEqual(
            fa.apply_fixed_selection_rules(base)["selection_status"],
            "pass",
        )

        one_core_failure = base.copy()
        one_core_failure["adjusted_rank_ic"] = 0.0149
        self.assertEqual(
            fa.apply_fixed_selection_rules(one_core_failure)[
                "selection_status"
            ],
            "watch",
        )

        all_core_failures = base.copy()
        all_core_failures["adjusted_rank_ic"] = 0.0149
        all_core_failures["adjusted_rank_icir"] = 0.19
        all_core_failures["adjusted_rank_ic_win_rate"] = 0.51
        all_core_failures["adjusted_positive_rank_ic_years"] = 2
        self.assertEqual(
            fa.apply_fixed_selection_rules(all_core_failures)[
                "selection_status"
            ],
            "reject",
        )

        low_valid_ratio = base.copy()
        low_valid_ratio["valid_rank_ic_day_ratio"] = 0.5999
        self.assertEqual(
            fa.apply_fixed_selection_rules(low_valid_ratio)[
                "selection_status"
            ],
            "reject",
        )

        outlier_concentrated = base.copy()
        outlier_concentrated["rank_ic_top_5pct_abs_share"] = 0.3001
        self.assertEqual(
            fa.apply_fixed_selection_rules(outlier_concentrated)[
                "selection_status"
            ],
            "watch",
        )

        failed = base.copy()
        failed["analysis_status"] = "no_valid_ic"
        self.assertEqual(
            fa.apply_fixed_selection_rules(failed)["selection_status"],
            "reject",
        )


if __name__ == "__main__":
    unittest.main()
