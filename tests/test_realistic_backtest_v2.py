"""现实约束缓冲持仓策略V2的行为测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import realistic_backtest as bt  # noqa: E402


def synthetic_inputs() -> dict[str, object]:
    dates = pd.bdate_range("2024-01-02", periods=36)
    codes = [f"S{i:03d}" for i in range(100)]
    values = np.arange(100.0)
    factor = pd.DataFrame(
        np.tile(values, (len(dates), 1)), index=dates, columns=codes
    )
    qfq_open = pd.DataFrame(10.0, index=dates, columns=codes)
    qfq_close = qfq_open.copy()
    raw_open = qfq_open.copy()
    amount = pd.DataFrame(100_000_000.0, index=dates, columns=codes)
    market_cap = pd.DataFrame(10_000_000_000.0, index=dates, columns=codes)
    up = pd.DataFrame(11.0, index=dates, columns=codes)
    down = pd.DataFrame(9.0, index=dates, columns=codes)
    suspended = pd.DataFrame(False, index=dates, columns=codes)
    membership = pd.DataFrame(True, index=dates, columns=codes)

    # 500万元股票目标恰好对应4.5手，V2 ROUND_HALF_UP应取5手。
    if_open = 5_000_000.0 / (4.5 * 300.0)
    basic = pd.DataFrame(
        {
            "ts_code": ["IF2406.CFX", "IF2407.CFX"],
            "multiplier": [300.0, 300.0],
            "list_date": [dates[0], dates[0]],
            "delist_date": [dates[-2], dates[-1] + pd.Timedelta(days=90)],
            "last_ddate": [dates[-2], dates[-1] + pd.Timedelta(days=90)],
        }
    )
    futures_daily = pd.DataFrame(
        [
            {
                "ts_code": contract,
                "trade_date": date,
                "pre_settle": if_open,
                "open": if_open,
                "settle": if_open,
                "vol": 100_000.0,
            }
            for date in dates
            for contract in basic["ts_code"]
        ]
    )
    futures_settle = pd.DataFrame(
        [
            {
                "ts_code": contract,
                "trade_date": dates[0],
                "short_margin_rate": 12.0,
                "offset_today_fee": 1000.0,
            }
            for contract in basic["ts_code"]
        ]
    )
    return locals()


def run_v2(
    inputs: dict[str, object],
    end_offset: int,
    *,
    rebalance_interval: int = 1,
    factor_signal_lag: int = 1,
    execution_price_field: str = "open",
    max_signal_replacements_per_day: int | None = None,
) -> dict[str, object]:
    dates = inputs["dates"]
    config = bt.BacktestConfig(
        strategy_mode="buffered_v2",
        target_stock_count=60,
        entry_fraction=0.15,
        retention_fraction=0.25,
        factor_missing_exit_days=5,
        stock_exposure_lower=0.40,
        stock_exposure_upper=0.60,
        single_stock_weight_cap=0.015,
        futures_contract_rounding="round_half_up",
        rebalance_interval=rebalance_interval,
        stock_impact_coefficient_bp=0.0,
        futures_impact_coefficient_bp=0.0,
    )
    return bt.backtest_single_factor(
        inputs["factor"],
        inputs["qfq_open"],
        inputs["raw_open"],
        inputs["amount"],
        inputs["market_cap"],
        inputs["up"],
        inputs["down"],
        inputs["suspended"],
        inputs["membership"],
        inputs["basic"],
        inputs["futures_daily"],
        inputs["futures_settle"],
        factor_name="alpha_v2_test",
        selected_direction=1.0,
        start_date=dates[21],
        end_date=dates[end_offset],
        config=config,
        qfq_close=inputs["qfq_close"],
        raw_close=inputs["qfq_close"],
        factor_signal_lag=factor_signal_lag,
        execution_price_field=execution_price_field,
        max_signal_replacements_per_day=max_signal_replacements_per_day,
    )


class RealisticBacktestV2Test(unittest.TestCase):
    def test_ndrop_limits_only_normal_replacement_pairs(self) -> None:
        inputs = synthetic_inputs()
        dates = inputs["dates"]
        factor = inputs["factor"]

        # Initial holdings are S040-S099. The next signal reverses the ranking,
        # producing 15 eligible exit-entry pairs before the NDrop limit.
        factor.loc[dates[21], :] = -np.arange(100.0)
        result = run_v2(
            inputs,
            24,
            max_signal_replacements_per_day=5,
        )
        day = result["daily"].loc[dates[22]]
        orders = result["orders"]
        normal = orders.loc[
            orders["date"].eq(dates[22])
            & orders["trigger_reason"].eq("normal_replacement")
        ]

        self.assertEqual(day["normal_exit_candidates_before_ndrop"], 60)
        self.assertEqual(day["normal_replacement_requests_before_ndrop"], 15)
        self.assertEqual(day["normal_replacements_blocked_by_ndrop"], 10)
        self.assertEqual(len(normal.loc[normal["side"].eq("sell")]), 5)
        self.assertEqual(len(normal.loc[normal["side"].eq("buy")]), 5)

    def test_ndrop_does_not_limit_mandatory_factor_missing_exit(self) -> None:
        inputs = synthetic_inputs()
        dates = inputs["dates"]
        inputs["factor"].loc[dates[21] : dates[25], "S098"] = np.nan

        result = run_v2(
            inputs,
            28,
            max_signal_replacements_per_day=5,
        )
        orders = result["orders"]
        forced = orders.loc[
            orders["date"].eq(dates[26])
            & orders["code"].eq("S098")
            & orders["side"].eq("sell")
        ]
        self.assertEqual(len(forced), 1)
        self.assertEqual(forced.iloc[0]["trigger_reason"], "factor_missing_5d")

    def test_ndrop_rejects_invalid_values(self) -> None:
        inputs = synthetic_inputs()
        with self.assertRaisesRegex(ValueError, "positive integer"):
            run_v2(inputs, 24, max_signal_replacements_per_day=0)

    def test_delay0_close_execution_uses_same_day_signal_and_next_close_return(self) -> None:
        inputs = synthetic_inputs()
        dates = inputs["dates"]
        inputs["qfq_close"].loc[dates[22], :] = 11.0
        inputs["futures_daily"].loc[:, "close"] = inputs["futures_daily"]["open"]

        result = run_v2(
            inputs,
            23,
            factor_signal_lag=0,
            execution_price_field="close",
        )
        daily = result["daily"]

        self.assertEqual(daily.loc[dates[21], "signal_date"], dates[21])
        self.assertEqual(daily.loc[dates[21], "execution_price_field"], "close")
        self.assertAlmostEqual(
            daily.loc[dates[22], "stock_market_pnl"],
            5_000_000.0 * 0.10,
            places=6,
        )
        first_holdings = result["holdings"].loc[
            result["holdings"]["date"].eq(dates[21])
        ]
        self.assertTrue(first_holdings["bought_today_notional"].gt(0).any())
        self.assertTrue(
            first_holdings["holding_period_market_pnl"].abs().lt(1e-8).all()
        )

    def test_delayed_signal_uses_global_calendar_lag(self) -> None:
        inputs = synthetic_inputs()
        dates = inputs["dates"]
        inputs["factor"].loc[dates[16], "S000"] = 10_000.0
        inputs["factor"].loc[dates[20], "S001"] = 20_000.0

        result = run_v2(inputs, 23, factor_signal_lag=5)
        first_day = result["daily"].loc[dates[21]]
        first_holdings = result["holdings"].loc[
            result["holdings"]["date"].eq(dates[21])
        ]

        self.assertEqual(first_day["signal_date"], dates[16])
        self.assertEqual(first_day["execution_info_date"], dates[20])
        self.assertEqual(first_day["factor_signal_lag"], 5)
        self.assertIn("S000", set(first_holdings["code"]))
        self.assertNotIn("S001", set(first_holdings["code"]))

    def test_nonscheduled_day_does_not_use_new_rank_for_normal_replacement(self) -> None:
        inputs = synthetic_inputs()
        dates = inputs["dates"]
        inputs["factor"].loc[dates[21], "S000"] = 10_000.0
        inputs["factor"].loc[dates[21], "S099"] = -10_000.0

        result = run_v2(inputs, 25, rebalance_interval=3)
        daily = result["daily"]
        orders = result["orders"]
        day_22_orders = orders.loc[orders["date"].eq(dates[22])]

        self.assertFalse(daily.loc[dates[22], "is_scheduled_rebalance"])
        self.assertFalse(
            day_22_orders["trigger_reason"].isin(
                ["normal_replacement", "vacancy_fill", "initial_entry"]
            ).any()
        )
        self.assertTrue(daily.loc[dates[24], "is_scheduled_rebalance"])

    def test_rank_and_if_half_up_helpers(self) -> None:
        signal = pd.Series({"B": 1.0, "A": 1.0, "C": 2.0})
        membership = pd.Series(True, index=signal.index)
        codes, ranks = bt.rank_valid_signal(signal, membership)

        self.assertEqual(codes, ["C", "A", "B"])
        self.assertEqual(ranks.to_dict(), {"C": 1, "A": 2, "B": 3})
        self.assertEqual(bt.round_target_contracts(4.49, "round_half_up"), 4)
        self.assertEqual(bt.round_target_contracts(4.50, "round_half_up"), 5)

    def test_initial_top60_and_blocked_sell_cancels_paired_buy(self) -> None:
        inputs = synthetic_inputs()
        dates = inputs["dates"]
        factor = inputs["factor"]

        # signal日让未持仓S039进入Top15，并让旧持仓S099跌到最差。
        factor.loc[dates[21], "S039"] = 1_000.0
        factor.loc[dates[21], "S099"] = -1_000.0
        # 下一开盘S099跌停，配对卖出失败，S039买单必须取消。
        inputs["raw_open"].loc[dates[22], "S099"] = 9.0
        inputs["qfq_open"].loc[dates[22], "S099"] = 9.0

        result = run_v2(inputs, 24)
        daily = result["daily"]
        first = daily.loc[dates[21]]
        self.assertTrue(first["initial_construction"])
        self.assertEqual(first["actual_stock_count"], 60)
        self.assertAlmostEqual(first["raw_target_if_contracts"], 4.5, places=6)
        self.assertEqual(first["target_short_contracts"], 5)

        orders = result["orders"]
        sell = orders.loc[
            orders["date"].eq(dates[22])
            & orders["code"].eq("S099")
            & orders["side"].eq("sell")
        ].iloc[0]
        buy = orders.loc[
            orders["date"].eq(dates[22])
            & orders["code"].eq("S039")
            & orders["side"].eq("buy")
        ].iloc[0]
        self.assertEqual(sell["unfilled_reason"], "open_at_down_limit")
        self.assertEqual(buy["unfilled_reason"], "paired_sell_not_completed")
        self.assertTrue(buy["paired_buy_cancelled_due_to_failed_sell"])
        self.assertEqual(daily.loc[dates[22], "actual_stock_count"], 60)

    def test_fifth_missing_signal_forces_unpaired_exit(self) -> None:
        inputs = synthetic_inputs()
        dates = inputs["dates"]
        factor = inputs["factor"]
        factor.loc[dates[21] : dates[25], "S098"] = np.nan

        result = run_v2(inputs, 28)
        orders = result["orders"]
        forced = orders.loc[
            orders["date"].eq(dates[26])
            & orders["code"].eq("S098")
            & orders["side"].eq("sell")
        ]
        self.assertEqual(len(forced), 1)
        self.assertEqual(forced.iloc[0]["trigger_reason"], "factor_missing_5d")
        self.assertEqual(forced.iloc[0]["filled_notional"], forced.iloc[0]["requested_notional"])

    def test_single_name_cap_trims_to_frozen_limit(self) -> None:
        inputs = synthetic_inputs()
        dates = inputs["dates"]
        inputs["qfq_close"].loc[dates[21], "S099"] = 40.0
        inputs["qfq_open"].loc[dates[22], "S099"] = 40.0
        inputs["raw_open"].loc[dates[22], "S099"] = 40.0
        inputs["up"].loc[dates[22], "S099"] = 44.0
        inputs["down"].loc[dates[22], "S099"] = 36.0

        result = run_v2(inputs, 24)
        orders = result["orders"]
        cap_order = orders.loc[
            orders["date"].eq(dates[22])
            & orders["code"].eq("S099")
            & orders["side"].eq("sell")
            & orders["trigger_reason"].eq("single_name_cap")
        ]
        self.assertEqual(len(cap_order), 1)
        holdings = result["holdings"]
        position = holdings.loc[
            holdings["date"].eq(dates[22]) & holdings["code"].eq("S099")
        ].iloc[0]
        self.assertLessEqual(position["posttrade_weight"], 0.0151)

    def test_total_exposure_above_sixty_percent_recovers_toward_fifty(self) -> None:
        inputs = synthetic_inputs()
        dates = inputs["dates"]
        inputs["qfq_close"].loc[dates[21], :] = 20.0
        inputs["qfq_open"].loc[dates[22], :] = 20.0
        inputs["raw_open"].loc[dates[22], :] = 20.0
        inputs["up"].loc[dates[22], :] = 22.0
        inputs["down"].loc[dates[22], :] = 18.0

        result = run_v2(inputs, 24)
        day = result["daily"].loc[dates[22]]
        self.assertGreater(day["signal_close_stock_exposure"], 0.60)
        self.assertTrue(day["stock_exposure_band_breached"])
        self.assertAlmostEqual(day["stock_exposure"], 0.50, delta=0.01)
        orders = result["orders"]
        exposure_sells = orders.loc[
            orders["date"].eq(dates[22])
            & orders["side"].eq("sell")
            & orders["trigger_reason"].eq("total_exposure_band")
        ]
        self.assertEqual(len(exposure_sells), 60)


if __name__ == "__main__":
    unittest.main()
