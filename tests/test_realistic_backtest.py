from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import realistic_backtest as bt  # noqa: E402


class RealisticBacktestUnitTest(unittest.TestCase):
    def test_stock_fees_are_directional_and_rounded(self) -> None:
        config = bt.BacktestConfig()
        buy = bt.calculate_stock_fees(100_000, "buy", pd.Timestamp("2023-09-01"), config)
        sell = bt.calculate_stock_fees(100_000, "sell", pd.Timestamp("2023-09-01"), config)

        self.assertEqual(buy["commission"], 20.0)
        self.assertEqual(buy["stamp_tax"], 0.0)
        self.assertEqual(sell["stamp_tax"], 50.0)
        self.assertGreater(sell["total_transaction_cost"], buy["total_transaction_cost"])

    def test_stock_fill_respects_limit_adv_and_market_cap(self) -> None:
        config = bt.BacktestConfig(stock_impact_coefficient_bp=0.0)
        fill, execution_price, reason, _ = bt.calculate_stock_fill(
            requested_notional=1_000_000,
            side="buy",
            open_price=10.0,
            up_limit=11.0,
            down_limit=9.0,
            adv_notional=10_000_000,
            current_notional=0.0,
            max_holding_notional=300_000.0,
            can_trade=True,
            config=config,
        )
        self.assertEqual(fill, 300_000.0)
        self.assertEqual(reason, "market_cap_capacity")
        self.assertGreater(execution_price, 10.0)

        blocked = bt.calculate_stock_fill(
            requested_notional=100_000,
            side="buy",
            open_price=11.0,
            up_limit=11.0,
            down_limit=9.0,
            adv_notional=10_000_000,
            current_notional=0.0,
            max_holding_notional=1_000_000.0,
            can_trade=True,
            config=config,
        )
        self.assertEqual(blocked[0], 0.0)
        self.assertEqual(blocked[2], "open_at_up_limit")

    def test_valid_observation_rolling_skips_suspension_gap(self) -> None:
        dates = pd.date_range("2025-01-01", periods=6)
        panel = pd.DataFrame({"A": [1.0, 2.0, np.nan, 3.0, 4.0, 5.0]}, index=dates)
        result = bt.rolling_valid_observation_mean(panel, 3)

        self.assertTrue(np.isnan(result.loc[dates[2], "A"]))
        self.assertEqual(result.loc[dates[3], "A"], 2.0)
        self.assertEqual(result.loc[dates[4], "A"], 3.0)

    def test_return_summary_includes_first_day_cost(self) -> None:
        returns = pd.Series([-0.01, 0.02], index=pd.date_range("2025-01-01", periods=2))
        nav = pd.Series([99.0, 100.98], index=returns.index)

        summary = bt._summary(returns, nav)

        self.assertAlmostEqual(summary["final_net_value"], 0.99 * 1.02)
        self.assertAlmostEqual(summary["cumulative_return"], 0.99 * 1.02 - 1)

    def test_contract_roll_starts_five_trading_days_before_delist(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=12)
        basic = pd.DataFrame(
            {
                "ts_code": ["IF1", "IF2"],
                "list_date": [dates[0], dates[0]],
                "delist_date": [dates[10], dates[-1] + pd.Timedelta(days=30)],
            }
        )
        daily = pd.DataFrame(
            [
                {"trade_date": date, "ts_code": code, "open": 4000.0}
                for date in dates
                for code in ["IF1", "IF2"]
            ]
        )
        config = bt.BacktestConfig(roll_days_before_delist=5)
        roll_start = dates[5]

        self.assertEqual(
            bt.choose_target_contract(dates[4], basic, daily, dates, config), "IF1"
        )
        self.assertEqual(
            bt.choose_target_contract(roll_start, basic, daily, dates, config), "IF2"
        )


class RealisticBacktestIntegrationTest(unittest.TestCase):
    def test_signal_executes_next_open_with_stock_and_if_books(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=32)
        codes = [f"S{i:03d}" for i in range(40)]
        factor = pd.DataFrame(
            np.tile(np.arange(40.0), (len(dates), 1)),
            index=dates,
            columns=codes,
        )
        qfq_open = pd.DataFrame(10.0, index=dates, columns=codes)
        raw_open = qfq_open.copy()
        amount = pd.DataFrame(100_000_000.0, index=dates, columns=codes)
        market_cap = pd.DataFrame(10_000_000_000.0, index=dates, columns=codes)
        up = pd.DataFrame(11.0, index=dates, columns=codes)
        down = pd.DataFrame(5.0, index=dates, columns=codes)
        suspended = pd.DataFrame(False, index=dates, columns=codes)
        membership = pd.DataFrame(True, index=dates, columns=codes)

        basic = pd.DataFrame(
            {
                "ts_code": ["IF2406.CFX", "IF2407.CFX"],
                "multiplier": [300.0, 300.0],
                "list_date": [dates[0], dates[0]],
                "delist_date": [dates[27], dates[-1] + pd.Timedelta(days=90)],
                "last_ddate": [dates[27], dates[-1] + pd.Timedelta(days=90)],
            }
        )
        futures_daily = pd.DataFrame(
            [
                {
                    "ts_code": contract,
                    "trade_date": date,
                    "pre_settle": 4000.0,
                    "open": 4000.0,
                    "settle": 4000.0,
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
        config = bt.BacktestConfig(
            min_stocks=30,
            rebalance_interval=2,
            stock_impact_coefficient_bp=0.0,
            futures_impact_coefficient_bp=0.0,
        )
        # 第一次建仓后的非调仓日，股票总回报下跌30%；该日又恰好开始换月，
        # 因而需要先开新腿再平掉部分新腿，触发平今费分支。
        qfq_open.loc[dates[22]:, codes[-8:]] = 7.0
        raw_open.loc[dates[22]:, codes[-8:]] = 7.0
        result = bt.backtest_single_factor(
            factor,
            qfq_open,
            raw_open,
            amount,
            market_cap,
            up,
            down,
            suspended,
            membership,
            basic,
            futures_daily,
            futures_settle,
            factor_name="alpha_test",
            selected_direction=1.0,
            start_date=dates[21],
            end_date=dates[-1],
            config=config,
        )
        daily = result["daily"]

        first = daily.iloc[0]
        self.assertEqual(first["signal_date"], dates[20])
        self.assertTrue(first["rebalance_executed"])
        self.assertEqual(first["actual_stock_count"], 8)
        self.assertGreater(first["actual_short_contracts"], 0)
        self.assertLess(first["nav_after_open"], config.initial_capital)
        self.assertFalse(result["orders"].empty)
        self.assertFalse(result["trades"].empty)
        first_holdings = result["holdings"].loc[
            result["holdings"]["date"].eq(dates[21])
        ]
        self.assertGreater(first_holdings["t_plus_one_blocked_notional"].sum(), 0)
        self.assertTrue(
            np.allclose(
                first_holdings["posttrade_notional"],
                first_holdings["sellable_notional"]
                + first_holdings["t_plus_one_blocked_notional"],
            )
        )
        self.assertTrue(
            np.allclose(
                daily["portfolio_pnl"],
                daily["gross_market_pnl"] - daily["total_cost"],
            )
        )
        future_orders = result["orders"].loc[result["orders"]["asset_type"].eq("future")]
        self.assertGreater(future_orders["close_today_contracts"].fillna(0).sum(), 0)


if __name__ == "__main__":
    unittest.main()
