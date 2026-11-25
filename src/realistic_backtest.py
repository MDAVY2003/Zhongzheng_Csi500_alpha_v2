"""单因子现实约束回测引擎。

信号在交易日 t 收盘后形成，并在下一交易日开盘执行。股票使用名义金额账户，
公司行动通过复权开盘收益反映；成交、涨跌停和容量判断全部使用未复权口径。
IF 使用具体月份合约、整数手、逐日结算和历史保证金参数。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from math import ceil, floor, sqrt
from typing import Any

import numpy as np
import pandas as pd

from constituent_exit import value_exit
from short_veto import ShortVetoConfig, plan_independent_targets


ANNUALIZATION_FACTOR = 252
MONEY_QUANTUM = Decimal("0.01")


@dataclass(frozen=True)
class BacktestConfig:
    """第一版现实约束回测参数。"""

    initial_capital: float = 10_000_000.0
    target_stock_exposure: float = 0.50
    quantile: float = 0.20
    min_stocks: int = 30
    rebalance_interval: int = 1
    stock_adv_window: int = 20
    stock_participation_rate: float = 0.05
    circulating_market_cap_rate: float = 0.001
    commission_rate: float = 2.0 / 10_000
    minimum_commission: float = 5.0
    stock_fixed_slippage_bp: float = 5.0
    stock_impact_coefficient_bp: float = 20.0
    futures_fixed_slippage_points: float = 0.4
    futures_impact_coefficient_bp: float = 10.0
    futures_participation_rate: float = 0.01
    futures_min_adv_observations: int = 5
    futures_adv_window: int = 20
    futures_exchange_fee_rate: float = 0.23 / 10_000
    futures_broker_fee_multiplier: float = 1.10
    futures_submission_fee: float = 1.0
    broker_margin_addon: float = 0.03
    futures_limit_rate: float = 0.10
    futures_tick_size: float = 0.2
    roll_days_before_delist: int = 5
    price_tolerance: float = 0.005
    strategy_mode: str = "full_rebalance"
    target_stock_count: int = 60
    entry_fraction: float = 0.15
    retention_fraction: float = 0.25
    factor_missing_exit_days: int = 5
    stock_exposure_lower: float = 0.40
    stock_exposure_upper: float = 0.60
    single_stock_weight_cap: float = 0.015
    futures_contract_rounding: str = "floor"

    idealized_constituent_exit: bool = False

    def validate(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital必须为正数")
        if not 0 < self.target_stock_exposure <= 1:
            raise ValueError("target_stock_exposure必须介于0和1之间")
        if not 0 < self.quantile <= 1:
            raise ValueError("quantile必须介于0和1之间")
        if self.min_stocks < 2:
            raise ValueError("min_stocks必须大于等于2")
        if self.rebalance_interval < 1:
            raise ValueError("rebalance_interval必须大于等于1")
        if self.stock_adv_window < 1 or self.futures_adv_window < 1:
            raise ValueError("ADV窗口必须为正整数")
        if self.strategy_mode not in {"full_rebalance", "buffered_v2"}:
            raise ValueError("strategy_mode必须是full_rebalance或buffered_v2")
        if self.target_stock_count < 1:
            raise ValueError("target_stock_count必须为正整数")
        if not 0 < self.entry_fraction < self.retention_fraction <= 1:
            raise ValueError("买入阈值必须小于保留阈值，且二者均介于0和1之间")
        if self.factor_missing_exit_days < 1:
            raise ValueError("factor_missing_exit_days必须为正整数")
        if not 0 <= self.stock_exposure_lower < self.target_stock_exposure:
            raise ValueError("股票敞口下限必须低于目标敞口")
        if not self.target_stock_exposure < self.stock_exposure_upper <= 1:
            raise ValueError("股票敞口上限必须高于目标敞口")
        if not 0 < self.single_stock_weight_cap <= 1:
            raise ValueError("single_stock_weight_cap必须介于0和1之间")
        if self.futures_contract_rounding not in {"floor", "round_half_up"}:
            raise ValueError("futures_contract_rounding必须是floor或round_half_up")


class NegativeCashError(RuntimeError):
    """现金变为负数时携带可落盘的失败快照。"""

    def __init__(self, message: str, snapshot: dict[str, Any]) -> None:
        super().__init__(message)
        self.snapshot = snapshot


def round_money(value: float) -> float:
    """按人民币分使用 ROUND_HALF_UP 舍入。"""
    return float(Decimal(str(value)).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP))


def stock_fee_rates(date: pd.Timestamp) -> dict[str, float]:
    """返回指定交易日适用的股票税费率。"""
    stamp_tax = 0.001 if date < pd.Timestamp("2023-08-28") else 0.0005
    transfer = 0.00002 if date < pd.Timestamp("2022-04-29") else 0.00001
    handling = 0.0000487 if date < pd.Timestamp("2023-08-28") else 0.0000341
    return {
        "stamp_tax": stamp_tax,
        "transfer_fee": transfer,
        "handling_fee": handling,
        "regulatory_fee": 0.00002,
    }


def calculate_stock_fees(
    notional: float,
    side: str,
    date: pd.Timestamp,
    config: BacktestConfig,
) -> dict[str, float]:
    """逐项计算并分币舍入一笔股票成交费用。"""
    if side not in {"buy", "sell"}:
        raise ValueError("股票成交方向必须是buy或sell")
    if notional <= 0:
        return {
            "commission": 0.0,
            "stamp_tax": 0.0,
            "transfer_fee": 0.0,
            "handling_fee": 0.0,
            "regulatory_fee": 0.0,
            "other_fee": 0.0,
            "total_transaction_cost": 0.0,
        }
    rates = stock_fee_rates(date)
    commission = round_money(max(config.minimum_commission, notional * config.commission_rate))
    stamp_tax = round_money(notional * rates["stamp_tax"]) if side == "sell" else 0.0
    transfer_fee = round_money(notional * rates["transfer_fee"])
    handling_fee = round_money(notional * rates["handling_fee"])
    regulatory_fee = round_money(notional * rates["regulatory_fee"])
    other_fee = round_money(handling_fee + regulatory_fee)
    total = round_money(commission + stamp_tax + transfer_fee + other_fee)
    return {
        "commission": commission,
        "stamp_tax": stamp_tax,
        "transfer_fee": transfer_fee,
        "handling_fee": handling_fee,
        "regulatory_fee": regulatory_fee,
        "other_fee": other_fee,
        "total_transaction_cost": total,
    }


def select_top_group(
    signal: pd.Series,
    membership: pd.Series,
    *,
    min_stocks: int,
    quantile: float,
) -> tuple[list[str], int]:
    """在信号日历史股票池内稳定选择方向调整后最高组。"""
    valid = pd.to_numeric(signal.where(membership.fillna(False)), errors="coerce")
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna()
    valid_count = len(valid)
    if valid_count < min_stocks:
        return [], valid_count
    ordered = pd.DataFrame(
        {"code": valid.index.astype(str), "value": valid.to_numpy(dtype=float)}
    ).sort_values(["value", "code"], kind="mergesort")
    group_size = max(1, int(floor(valid_count * quantile)))
    return ordered.iloc[-group_size:]["code"].tolist(), valid_count


def rank_valid_signal(
    signal: pd.Series,
    membership: pd.Series,
) -> tuple[list[str], pd.Series]:
    """按因子值降序、代码升序生成唯一横截面序号。"""
    valid = pd.to_numeric(signal.where(membership.fillna(False)), errors="coerce")
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna()
    ordered = pd.DataFrame(
        {"code": valid.index.astype(str), "value": valid.to_numpy(dtype=float)}
    ).sort_values(["value", "code"], ascending=[False, True], kind="mergesort")
    codes = ordered["code"].tolist()
    ranks = pd.Series(
        np.arange(1, len(codes) + 1, dtype=int),
        index=pd.Index(codes),
        dtype="int64",
    )
    return codes, ranks


def round_target_contracts(raw_contracts: float, method: str) -> int:
    """将非负IF目标手数按冻结规则转换为整数手。"""
    if not np.isfinite(raw_contracts) or raw_contracts <= 0:
        return 0
    if method == "floor":
        return floor(raw_contracts)
    if method == "round_half_up":
        return floor(raw_contracts + 0.5)
    raise ValueError(f"未知IF手数取整规则：{method}")


def rolling_valid_observation_mean(
    panel: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """按每只股票最近 window 个有限正值计算均值，并只向未来沿用。"""
    result = pd.DataFrame(np.nan, index=panel.index, columns=panel.columns)
    for column in panel.columns:
        series = pd.to_numeric(panel[column], errors="coerce")
        valid = series[series.gt(0) & np.isfinite(series)]
        rolled = valid.rolling(window, min_periods=window).mean()
        result[column] = rolled.reindex(panel.index).ffill()
    return result


def calculate_stock_fill(
    *,
    requested_notional: float,
    side: str,
    open_price: float,
    up_limit: float,
    down_limit: float,
    adv_notional: float,
    current_notional: float,
    max_holding_notional: float | None,
    can_trade: bool,
    config: BacktestConfig,
    price_field: str = "open",
) -> tuple[float, float, str, float]:
    """计算股票订单可成交金额、执行价、未成交原因和冲击bp。"""
    if requested_notional <= 0:
        return 0.0, np.nan, "no_order", 0.0
    if side not in {"buy", "sell"}:
        raise ValueError("side必须是buy或sell")
    if price_field not in {"open", "close"}:
        raise ValueError("price_field必须是open或close")
    if not can_trade or not np.isfinite(open_price) or open_price <= 0:
        return 0.0, np.nan, f"missing_{price_field}_or_suspended", 0.0
    if not np.isfinite(adv_notional) or adv_notional <= 0:
        return 0.0, np.nan, "insufficient_adv_history", 0.0
    if side == "buy" and np.isfinite(up_limit) and open_price >= up_limit - config.price_tolerance:
        return 0.0, np.nan, f"{price_field}_at_up_limit", 0.0
    if side == "sell" and np.isfinite(down_limit) and open_price <= down_limit + config.price_tolerance:
        return 0.0, np.nan, f"{price_field}_at_down_limit", 0.0

    fill = min(requested_notional, adv_notional * config.stock_participation_rate)
    reason = "filled"
    if fill < requested_notional:
        reason = "adv_capacity"
    if side == "buy":
        if max_holding_notional is None or not np.isfinite(max_holding_notional):
            return 0.0, np.nan, "invalid_circulating_market_cap", 0.0
        holding_room = max(0.0, max_holding_notional - current_notional)
        if holding_room <= 0:
            return 0.0, np.nan, "market_cap_capacity", 0.0
        if holding_room < fill:
            fill = holding_room
            reason = "market_cap_capacity"

    # 若滑点后的价格会越过涨跌停价，反解允许的最大成交金额。
    limit_room_bp = np.inf
    if side == "buy" and np.isfinite(up_limit):
        limit_room_bp = max(0.0, (up_limit / open_price - 1.0) * 10_000)
    elif side == "sell" and np.isfinite(down_limit):
        limit_room_bp = max(0.0, (1.0 - down_limit / open_price) * 10_000)
    if np.isfinite(limit_room_bp):
        impact_room = limit_room_bp - config.stock_fixed_slippage_bp
        if impact_room <= 0:
            return 0.0, np.nan, "slippage_crosses_limit", 0.0
        if config.stock_impact_coefficient_bp > 0:
            max_by_limit = adv_notional * (
                impact_room / config.stock_impact_coefficient_bp
            ) ** 2
            if max_by_limit < fill:
                fill = max(0.0, max_by_limit)
                reason = "slippage_limit_capacity"
    if fill <= 1e-8:
        return 0.0, np.nan, reason, 0.0

    impact_bp = config.stock_impact_coefficient_bp * sqrt(fill / adv_notional)
    total_bp = config.stock_fixed_slippage_bp + impact_bp
    direction = 1.0 if side == "buy" else -1.0
    execution_price = open_price * (1.0 + direction * total_bp / 10_000)
    if fill + 1e-8 < requested_notional and reason == "filled":
        reason = "partial_fill"
    return float(fill), float(execution_price), reason, float(impact_bp)


def _asof_contract_parameter(
    parameters: pd.DataFrame,
    contract: str,
    date: pd.Timestamp,
    field: str,
) -> float:
    rows = parameters.loc[
        parameters["ts_code"].eq(contract)
        & parameters["trade_date"].le(date),
        ["trade_date", field],
    ].dropna(subset=[field])
    if rows.empty:
        return np.nan
    return float(rows.sort_values("trade_date").iloc[-1][field])


def _contract_row(
    futures_daily: pd.DataFrame,
    contract: str,
    date: pd.Timestamp,
) -> pd.Series | None:
    rows = futures_daily.loc[
        futures_daily["ts_code"].eq(contract)
        & futures_daily["trade_date"].eq(date)
    ]
    return None if rows.empty else rows.iloc[0]


def _futures_adv(
    futures_daily: pd.DataFrame,
    contract: str,
    signal_date: pd.Timestamp,
    config: BacktestConfig,
) -> tuple[float, int]:
    rows = futures_daily.loc[
        futures_daily["ts_code"].eq(contract)
        & futures_daily["trade_date"].le(signal_date),
        ["trade_date", "vol"],
    ].sort_values("trade_date")
    valid = pd.to_numeric(rows["vol"], errors="coerce")
    valid = valid[np.isfinite(valid) & valid.gt(0)].tail(config.futures_adv_window)
    count = len(valid)
    if count < config.futures_min_adv_observations:
        return np.nan, count
    return float(valid.mean()), count


def _roll_start_date(
    delist_date: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
    days_before: int,
) -> pd.Timestamp:
    eligible = trading_dates[trading_dates <= delist_date]
    if len(eligible) <= days_before:
        return eligible[0]
    return eligible[-(days_before + 1)]


def choose_target_contract(
    date: pd.Timestamp,
    futures_basic: pd.DataFrame,
    futures_daily: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    config: BacktestConfig,
) -> str | None:
    """选择当月近月合约；进入换月窗口后选择下一到期合约。"""
    candidates = futures_basic.loc[
        futures_basic["list_date"].le(date)
        & futures_basic["delist_date"].ge(date)
    ].sort_values(["delist_date", "ts_code"])
    if candidates.empty:
        return None
    available = set(
        futures_daily.loc[
            futures_daily["trade_date"].eq(date)
            & futures_daily["open"].notna(),
            "ts_code",
        ]
    )
    candidates = candidates.loc[candidates["ts_code"].isin(available)]
    if candidates.empty:
        return None
    near = candidates.iloc[0]
    roll_start = _roll_start_date(
        pd.Timestamp(near["delist_date"]),
        trading_dates,
        config.roll_days_before_delist,
    )
    if date >= roll_start and len(candidates) > 1:
        return str(candidates.iloc[1]["ts_code"])
    return str(near["ts_code"])


def _futures_limits(
    row: pd.Series,
    config: BacktestConfig,
) -> tuple[float, float]:
    pre_settle = float(row["pre_settle"])
    tick = config.futures_tick_size
    up = floor((pre_settle * (1 + config.futures_limit_rate)) / tick) * tick
    down = np.ceil((pre_settle * (1 - config.futures_limit_rate)) / tick) * tick
    return float(up), float(down)


def _futures_trade_capacity(
    *,
    requested_contracts: int,
    side: str,
    row: pd.Series | None,
    futures_adv: float,
    used_capacity: int,
    config: BacktestConfig,
    price_field: str = "open",
) -> tuple[int, float, float, str]:
    """返回期货可成交手数、成交价、冲击bp和原因。"""
    if requested_contracts <= 0:
        return 0, np.nan, 0.0, "no_order"
    if price_field not in {"open", "close"}:
        raise ValueError("price_field必须是open或close")
    if row is None or not np.isfinite(row.get(price_field, np.nan)):
        return 0, np.nan, 0.0, f"missing_futures_{price_field}"
    if not np.isfinite(futures_adv) or futures_adv <= 0:
        return 0, np.nan, 0.0, "insufficient_futures_adv_history"
    open_price = float(row[price_field])
    up, down = _futures_limits(row, config)
    if side == "buy" and open_price >= up - config.futures_tick_size / 2:
        return 0, np.nan, 0.0, "futures_at_up_limit"
    if side == "sell" and open_price <= down + config.futures_tick_size / 2:
        return 0, np.nan, 0.0, "futures_at_down_limit"
    daily_capacity = max(0, floor(futures_adv * config.futures_participation_rate))
    fill = min(requested_contracts, max(0, daily_capacity - used_capacity))
    if fill <= 0:
        return 0, np.nan, 0.0, "futures_adv_capacity"

    impact_bp = config.futures_impact_coefficient_bp * sqrt(fill / futures_adv)
    impact_points = open_price * impact_bp / 10_000
    direction = 1.0 if side == "buy" else -1.0
    execution_price = open_price + direction * (
        config.futures_fixed_slippage_points + impact_points
    )
    if (side == "buy" and execution_price > up) or (
        side == "sell" and execution_price < down
    ):
        return 0, np.nan, 0.0, "futures_slippage_crosses_limit"
    reason = "filled" if fill == requested_contracts else "futures_adv_capacity"
    return int(fill), float(execution_price), float(impact_bp), reason


def _summary(returns: pd.Series, nav: pd.Series) -> pd.Series:
    valid = pd.to_numeric(returns, errors="coerce").dropna()
    if valid.empty:
        return pd.Series({"valid_days": 0})
    std = float(valid.std(ddof=1))
    final_nav = float((1.0 + valid).prod())
    annualized_return = (
        final_nav ** (ANNUALIZATION_FACTOR / len(valid)) - 1
        if final_nav > 0
        else np.nan
    )
    nav_curve = (1.0 + valid).cumprod()
    running_peak = pd.concat(
        [pd.Series([1.0]), nav_curve.reset_index(drop=True)],
        ignore_index=True,
    ).cummax().iloc[1:].set_axis(nav_curve.index)
    drawdown = nav_curve / running_peak - 1
    return pd.Series(
        {
            "valid_days": len(valid),
            "cumulative_return": final_nav - 1,
            "annualized_return": annualized_return,
            "annualized_volatility": std * sqrt(ANNUALIZATION_FACTOR),
            "annualized_sharpe": (
                float(valid.mean()) / std * sqrt(ANNUALIZATION_FACTOR)
                if std > 0
                else np.nan
            ),
            "win_rate": float(valid.gt(0).mean()),
            "final_net_value": final_nav,
            "max_drawdown": float(drawdown.min()),
        }
    )


def _yearly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, frame in daily.groupby(daily.index.year):
        metrics = _summary(frame["net_return"], frame["nav_after_open"])
        gross_nav = (1.0 + frame["gross_return_before_cost"].fillna(0.0)).cumprod()
        gross_metrics = _summary(frame["gross_return_before_cost"], gross_nav)
        rows.append(
            {
                "year": int(year),
                **metrics.to_dict(),
                **{f"gross_{key}": value for key, value in gross_metrics.items()},
                "mean_stock_exposure": frame["stock_exposure"].mean(),
                "mean_futures_short_exposure": frame["futures_short_exposure"].mean(),
                "mean_net_exposure": frame["net_exposure"].mean(),
                "mean_cash_ratio": frame["cash_balance"].div(frame["nav_after_open"]).mean(),
                "total_transaction_cost": frame["total_cost"].sum(),
            }
        )
    return pd.DataFrame(rows).set_index("year")


def backtest_single_factor(
    factor: pd.DataFrame,
    qfq_open: pd.DataFrame,
    raw_open: pd.DataFrame,
    amount: pd.DataFrame,
    circulating_market_cap: pd.DataFrame,
    up_limit: pd.DataFrame,
    down_limit: pd.DataFrame,
    full_day_suspension: pd.DataFrame,
    membership: pd.DataFrame,
    futures_basic: pd.DataFrame,
    futures_daily: pd.DataFrame,
    futures_settle: pd.DataFrame,
    *,
    factor_name: str,
    selected_direction: float,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    config: BacktestConfig | None = None,
    qfq_close: pd.DataFrame | None = None,
    raw_close: pd.DataFrame | None = None,
    precomputed_stock_adv: pd.DataFrame | None = None,
    precomputed_market_cap_avg: pd.DataFrame | None = None,
    factor_signal_lag: int = 1,
    execution_price_field: str = "open",
    max_signal_replacements_per_day: int | None = None,
    futures_hedge_enabled: bool = True,
    independent_stock_sets: bool = False,
    short_factor: pd.DataFrame | None = None,
    short_veto_config: ShortVetoConfig | None = None,
) -> dict[str, object]:
    """运行一个因子的现实约束样本外回测。"""
    cfg = config or BacktestConfig()
    cfg.validate()
    if independent_stock_sets and cfg.strategy_mode != "buffered_v2":
        raise ValueError("Independent sets require buffered_v2")
    if short_veto_config is not None and not independent_stock_sets:
        raise ValueError("Short veto requires independent_stock_sets")
    veto_policy = short_veto_config or ShortVetoConfig(enabled=False)
    veto_policy.validate()
    if veto_policy.enabled:
        if short_factor is None or not factor.index.equals(short_factor.index) or not factor.columns.equals(short_factor.columns):
            raise ValueError("Short signal must exactly align with Long signal")
    if cfg.idealized_constituent_exit and (qfq_close is None or raw_close is None or execution_price_field != "open"):
        raise ValueError("Idealized close exits require both close panels and normal open execution")
    if selected_direction not in {-1, 1, -1.0, 1.0}:
        raise ValueError("selected_direction必须是1或-1")
    if not isinstance(factor_signal_lag, int) or factor_signal_lag < 0:
        raise ValueError("factor_signal_lag必须是大于等于0的整数")
    if (
        max_signal_replacements_per_day is not None
        and (
            isinstance(max_signal_replacements_per_day, bool)
            or not isinstance(max_signal_replacements_per_day, int)
            or max_signal_replacements_per_day < 1
        )
    ):
        raise ValueError(
            "max_signal_replacements_per_day must be a positive integer or None"
        )
    if max_signal_replacements_per_day is not None and cfg.strategy_mode != "buffered_v2":
        raise ValueError("NDrop is only supported by buffered_v2")
    if execution_price_field not in {"open", "close"}:
        raise ValueError("execution_price_field必须是open或close")
    if execution_price_field == "close" and (qfq_close is None or raw_close is None):
        raise ValueError("close执行模式必须提供qfq_close和raw_close")
    panels = {
        "qfq_open": qfq_open,
        "raw_open": raw_open,
        "amount": amount,
        "circulating_market_cap": circulating_market_cap,
        "up_limit": up_limit,
        "down_limit": down_limit,
        "full_day_suspension": full_day_suspension,
        "membership": membership,
    }
    if cfg.strategy_mode == "buffered_v2":
        if qfq_close is None:
            raise ValueError("buffered_v2必须提供qfq_close")
        panels["qfq_close"] = qfq_close
    if execution_price_field == "close":
        panels["raw_close"] = raw_close
    for name, panel in panels.items():
        if not factor.index.equals(panel.index) or not factor.columns.equals(panel.columns):
            raise ValueError(f"factor与{name}的index或columns不一致")
    if not isinstance(factor.index, pd.DatetimeIndex) or not factor.index.is_monotonic_increasing:
        raise ValueError("factor必须使用升序DatetimeIndex")
    if factor.index.has_duplicates or factor.columns.has_duplicates:
        raise ValueError("factor的日期或股票代码存在重复")

    if not futures_hedge_enabled:
        # Truly unhedged: do not load/query/use any futures input or reserve margin.
        futures_basic = pd.DataFrame(columns=["ts_code","list_date","delist_date","last_ddate","multiplier"])
        futures_daily = pd.DataFrame(columns=["trade_date","ts_code","vol"])
        futures_settle = pd.DataFrame(columns=["trade_date","ts_code","short_margin_rate","offset_today_fee"])
    futures_basic = futures_basic.copy()
    futures_daily = futures_daily.copy().sort_values(["trade_date", "ts_code"])
    futures_settle = futures_settle.copy().sort_values(["trade_date", "ts_code"])
    for column in ["list_date", "delist_date", "last_ddate"]:
        futures_basic[column] = pd.to_datetime(futures_basic[column], errors="raise")
    futures_daily["trade_date"] = pd.to_datetime(futures_daily["trade_date"], errors="raise")
    futures_settle["trade_date"] = pd.to_datetime(futures_settle["trade_date"], errors="raise")
    futures_basic_lookup = futures_basic.set_index("ts_code")
    futures_row_lookup = futures_daily.set_index(["trade_date", "ts_code"])
    futures_volume_lookup = {
        str(contract): group.set_index("trade_date")["vol"].sort_index()
        for contract, group in futures_daily.groupby("ts_code", sort=False)
    }
    futures_parameter_lookup = {
        (str(contract), field): group.set_index("trade_date")[field]
        .dropna()
        .sort_index()
        for contract, group in futures_settle.groupby("ts_code", sort=False)
        for field in ["short_margin_rate", "offset_today_fee"]
    }

    def future_row(contract: str, date: pd.Timestamp) -> pd.Series | None:
        try:
            row = futures_row_lookup.loc[(date, contract)]
        except KeyError:
            return None
        if isinstance(row, pd.DataFrame):
            raise ValueError(f"IF日线存在重复记录：{date.date()} {contract}")
        return row

    def future_adv(contract: str, signal_date: pd.Timestamp) -> tuple[float, int]:
        series = futures_volume_lookup.get(contract)
        if series is None:
            return np.nan, 0
        valid = pd.to_numeric(series.loc[:signal_date], errors="coerce")
        valid = valid[np.isfinite(valid) & valid.gt(0)].tail(cfg.futures_adv_window)
        if len(valid) < cfg.futures_min_adv_observations:
            return np.nan, len(valid)
        return float(valid.mean()), len(valid)

    def future_parameter(contract: str, date: pd.Timestamp, field: str) -> float:
        series = futures_parameter_lookup.get((contract, field))
        if series is None:
            return np.nan
        available = series.loc[:date]
        return np.nan if available.empty else float(available.iloc[-1])

    def target_contract_for(date: pd.Timestamp) -> str | None:
        candidates = futures_basic.loc[
            futures_basic["list_date"].le(date)
            & futures_basic["delist_date"].ge(date)
        ].sort_values(["delist_date", "ts_code"])
        if candidates.empty:
            return None
        available_codes = [
            str(code)
            for code in candidates["ts_code"]
            if future_row(str(code), date) is not None
        ]
        if not available_codes:
            return None
        near = available_codes[0]
        near_delist = pd.Timestamp(futures_basic_lookup.at[near, "delist_date"])
        roll_start = _roll_start_date(
            near_delist,
            factor.index,
            cfg.roll_days_before_delist,
        )
        if date >= roll_start and len(available_codes) > 1:
            return available_codes[1]
        return near

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    dates = factor.index[(factor.index >= start) & (factor.index <= end)]
    if len(dates) < 2:
        raise ValueError("回测区间至少需要两个交易日")
    if factor.index.get_loc(dates[0]) < max(1, factor_signal_lag):
        raise ValueError("首个执行日前缺少所需的因子信号交易日")

    adjusted_factor = factor * float(selected_direction)
    stock_adv_panel = (
        rolling_valid_observation_mean(amount, cfg.stock_adv_window)
        if precomputed_stock_adv is None
        else precomputed_stock_adv
    )
    market_cap_avg_panel = (
        rolling_valid_observation_mean(circulating_market_cap, cfg.stock_adv_window)
        if precomputed_market_cap_avg is None
        else precomputed_market_cap_avg
    )
    for name, panel in {
        "precomputed_stock_adv": stock_adv_panel,
        "precomputed_market_cap_avg": market_cap_avg_panel,
    }.items():
        if not factor.index.equals(panel.index) or not factor.columns.equals(panel.columns):
            raise ValueError(f"factor与{name}的index或columns不一致")
    holdings = pd.Series(0.0, index=factor.columns, dtype=float)
    last_qfq = pd.Series(np.nan, index=factor.columns, dtype=float)
    stale_days = pd.Series(0, index=factor.columns, dtype=int)
    missing_factor_days = pd.Series(0, index=factor.columns, dtype=int)
    single_cap_breach_days = pd.Series(0, index=factor.columns, dtype=int)
    cash = float(cfg.initial_capital)
    futures_positions: dict[str, int] = {}
    futures_basis: dict[str, float] = {}
    previous_nav = float(cfg.initial_capital)
    previous_date: pd.Timestamp | None = None
    margin_call_pending = False

    daily_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    holding_rows: list[dict[str, Any]] = []
    futures_position_rows: list[dict[str, Any]] = []
    roll_rows: list[dict[str, Any]] = []
    exit_rows: list[dict[str, Any]] = []
    veto_events: list[dict[str, Any]] = []
    veto_age: dict[str, int] = {}
    pending_index_removals: set[str] = set()
    pending_exit_postings: list[dict[str, Any]] = []
    close_asof = qfq_close.where(qfq_close.gt(0)).ffill() if cfg.idealized_constituent_exit else None
    raw_close_asof = raw_close.where(raw_close.gt(0)).ffill() if cfg.idealized_constituent_exit else None

    for day_number, date in enumerate(dates):
        full_location = factor.index.get_loc(date)
        execution_info_date = factor.index[full_location - 1]
        factor_signal_date = factor.index[full_location - factor_signal_lag]
        execution_qfq_panel = qfq_open if execution_price_field == "open" else qfq_close
        execution_raw_panel = raw_open if execution_price_field == "open" else raw_close
        day_qfq_open = execution_qfq_panel.loc[date]
        day_raw_open = execution_raw_panel.loc[date]
        day_up_limit = up_limit.loc[date]
        day_down_limit = down_limit.loc[date]
        factor_membership = membership.loc[factor_signal_date]
        execution_membership = membership.loc[date]
        cash_at_start = cash
        stock_value_before = float(holdings.sum())

        # V2在信号日收盘检查风险，但正式账户仍只按开盘到开盘记账。
        signal_close_holdings = holdings.copy()
        if cfg.strategy_mode == "buffered_v2":
            signal_close_prices = qfq_close.loc[
                date if execution_price_field == "close" else execution_info_date
            ]
            for code in holdings.index[holdings.gt(1e-8)]:
                close_price = signal_close_prices[code]
                basis_price = last_qfq[code]
                if (
                    np.isfinite(close_price)
                    and close_price > 0
                    and np.isfinite(basis_price)
                    and basis_price > 0
                ):
                    signal_close_holdings[code] *= float(close_price / basis_price)
            signal_close_nav = cash + float(signal_close_holdings.sum())
            if signal_close_nav <= 0:
                raise RuntimeError(f"{execution_info_date.date()}收盘风险监控净资产非正")
        else:
            signal_close_nav = np.nan

        # 旧股票持仓按最近两个可得复权开盘价结算；停牌期间维持最近估值。
        stock_market_pnl = sum(item["market_pnl"] for item in pending_exit_postings)
        stock_pnl_by_code = pd.Series(0.0, index=factor.columns, dtype=float)
        for code in holdings.index[holdings.gt(1e-8)]:
            price = day_qfq_open[code]
            if np.isfinite(price) and price > 0:
                if np.isfinite(last_qfq[code]) and last_qfq[code] > 0:
                    old_value = holdings[code]
                    holdings[code] *= float(price / last_qfq[code])
                    code_pnl = float(holdings[code] - old_value)
                    stock_pnl_by_code[code] = code_pnl
                    stock_market_pnl += code_pnl
                last_qfq[code] = float(price)
                stale_days[code] = 0
            else:
                stale_days[code] += 1

        # 期货在当前开盘相对上一结算价的未实现价值计入开盘净值。
        futures_open_value_before = 0.0
        for contract, position in list(futures_positions.items()):
            row = future_row(contract, date)
            if row is not None and np.isfinite(row[execution_price_field]):
                multiplier = float(futures_basic_lookup.at[contract, "multiplier"])
                futures_open_value_before += position * multiplier * (
                    float(row[execution_price_field]) - futures_basis[contract]
                )
        nav_pretrade = cash + float(holdings.sum()) + futures_open_value_before
        if nav_pretrade <= 0:
            raise RuntimeError(f"{date.date()}组合净资产非正")
        pretrade_holdings = holdings.copy()

        scheduled = day_number % cfg.rebalance_interval == 0
        if cfg.idealized_constituent_exit:
            pending_index_removals.update(str(code) for code in holdings.index[holdings.gt(1e-8)] if not bool(execution_membership.get(code, False)))
        independent_buy_order: list[str] = []
        veto_diagnostics: dict[str, int] = {}
        sell_trigger: dict[str, str] = {}
        buy_trigger: dict[str, str] = {}
        sell_pair_id: dict[str, str] = {}
        buy_pair_id: dict[str, str] = {}
        buy_gate_pair: dict[str, str] = {}
        exact_ranks = pd.Series(dtype="int64")
        entry_codes: list[str] = []
        retention_codes: list[str] = []
        exposure_band_breached = False
        initial_construction = False
        desired = holdings.copy()
        signal_valid_count = 0
        normal_exit_candidates_before_ndrop = 0
        normal_replacement_requests_before_ndrop = 0
        normal_replacements_blocked_by_ndrop = 0

        if cfg.strategy_mode == "buffered_v2":
            ordered_codes: list[str] = []
            if scheduled:
                ordered_codes, exact_ranks = rank_valid_signal(
                    adjusted_factor.loc[factor_signal_date], factor_membership
                )
                signal_valid_count = len(ordered_codes)
                entry_count = ceil(signal_valid_count * cfg.entry_fraction)
                retention_count = ceil(signal_valid_count * cfg.retention_fraction)
                entry_codes = ordered_codes[:entry_count]
                retention_codes = ordered_codes[:retention_count]
            held_codes = holdings.index[holdings.gt(1e-8)].astype(str).tolist()
            held_set = set(held_codes)
            signal_is_valid = scheduled and signal_valid_count >= cfg.min_stocks

            mandatory_exits: list[str] = []
            signal_values = pd.Series(np.nan, index=factor.columns, dtype=float)
            if scheduled and signal_is_valid:
                signal_values = pd.to_numeric(
                    adjusted_factor.loc[factor_signal_date], errors="coerce"
                ).replace([np.inf, -np.inf], np.nan)
            for code in held_codes:
                if cfg.idealized_constituent_exit and code in pending_index_removals:
                    missing_factor_days[code] = 0
                    continue
                if not bool(execution_membership.get(code, False)):
                    missing_factor_days[code] = 0
                    mandatory_exits.append(code)
                elif scheduled and signal_is_valid:
                    if np.isfinite(signal_values.get(code, np.nan)):
                        missing_factor_days[code] = 0
                    else:
                        missing_factor_days[code] += 1
                        if missing_factor_days[code] >= cfg.factor_missing_exit_days:
                            mandatory_exits.append(code)
                elif missing_factor_days[code] >= cfg.factor_missing_exit_days:
                    mandatory_exits.append(code)

            can_form_initial = (
                scheduled
                and
                not held_codes
                and len([code for code in ordered_codes if bool(execution_membership.get(code, False))])
                >= cfg.target_stock_count
            )
            normal_exits: list[str] = []
            if scheduled and signal_is_valid and held_codes and not can_form_initial:
                retention_set = set(retention_codes)
                normal_exits = [
                    code
                    for code in held_codes
                    if code not in mandatory_exits and code not in pending_index_removals
                    and code in exact_ranks.index
                    and code not in retention_set
                ]
                normal_exits.sort(
                    key=lambda code: (int(exact_ranks.get(code, 10**9)), code),
                    reverse=True,
                )
            normal_exit_candidates_before_ndrop = len(normal_exits)

            if independent_stock_sets:
                initial_construction = can_form_initial
                planned = plan_independent_targets(
                    ordered_codes=ordered_codes, long_ranks=exact_ranks,
                    long_entries=entry_codes, normal_exits=normal_exits,
                    mandatory_exits=mandatory_exits, pending_exits=pending_index_removals,
                    held_codes=held_codes, close_holdings=signal_close_holdings,
                    close_nav=signal_close_nav, execution_membership=execution_membership,
                    short_values=None if short_factor is None else short_factor.loc[factor_signal_date],
                    short_membership=factor_membership, config=cfg, policy=veto_policy,
                    veto_age=veto_age, signal_valid=signal_is_valid, initial=can_form_initial,
                    date=date, info_date=execution_info_date, execute_allowed=day_number < len(dates)-1,
                    max_normal_replacements=max_signal_replacements_per_day)
                desired_close=planned["desired"]
                sell_trigger.update(planned["sells"]); buy_trigger.update(planned["buys"])
                independent_buy_order=planned["buy_order"]
                veto_events.extend(planned["events"]); veto_diagnostics=planned["diagnostics"]
            else:
                desired_close = signal_close_holdings.copy()
                desired_close.loc[mandatory_exits] = 0.0
                available_entries = [
                    code
                    for code in entry_codes
                    if code not in held_set and bool(execution_membership.get(code, False))
                ]
                base_notional = signal_close_nav * cfg.target_stock_exposure / cfg.target_stock_count

                if can_form_initial:
                    initial_construction = True
                    desired_close[:] = 0.0
                    initial_codes = [
                        code for code in ordered_codes if bool(execution_membership.get(code, False))
                    ][: cfg.target_stock_count]
                    for code in initial_codes:
                        desired_close[code] = base_notional
                        buy_trigger[code] = "initial_entry"
                elif signal_is_valid:
                    candidate_cursor = 0
                    existing_vacancies = max(0, cfg.target_stock_count - len(held_codes))
                    for _ in range(existing_vacancies):
                        if candidate_cursor >= len(available_entries):
                            break
                        code = available_entries[candidate_cursor]
                        candidate_cursor += 1
                        desired_close[code] = base_notional
                        buy_trigger[code] = "vacancy_fill"

                    for forced_index, exit_code in enumerate(mandatory_exits, start=1):
                        trigger = (
                            "index_removal"
                            if not bool(execution_membership.get(exit_code, False))
                            else "factor_missing_5d"
                        )
                        sell_trigger[exit_code] = trigger
                        if candidate_cursor < len(available_entries):
                            buy_code = available_entries[candidate_cursor]
                            candidate_cursor += 1
                            pair_id = f"{factor_signal_date.date()}-forced-{forced_index:03d}"
                            desired_close[buy_code] = base_notional
                            buy_trigger[buy_code] = "vacancy_fill"
                            sell_pair_id[exit_code] = pair_id
                            buy_pair_id[buy_code] = pair_id
                            buy_gate_pair[buy_code] = pair_id

                    remaining_entry_capacity = max(
                        0, len(available_entries) - candidate_cursor
                    )
                    normal_replacement_requests_before_ndrop = min(
                        len(normal_exits), remaining_entry_capacity
                    )
                    if max_signal_replacements_per_day is not None:
                        allowed_replacements = min(
                            normal_replacement_requests_before_ndrop,
                            max_signal_replacements_per_day,
                        )
                        normal_replacements_blocked_by_ndrop = (
                            normal_replacement_requests_before_ndrop
                            - allowed_replacements
                        )
                        normal_exits = normal_exits[:allowed_replacements]

                    for normal_index, exit_code in enumerate(normal_exits, start=1):
                        if candidate_cursor >= len(available_entries):
                            break
                        buy_code = available_entries[candidate_cursor]
                        candidate_cursor += 1
                        pair_id = f"{factor_signal_date.date()}-normal-{normal_index:03d}"
                        desired_close[exit_code] = 0.0
                        desired_close[buy_code] = base_notional
                        sell_trigger[exit_code] = "normal_replacement"
                        buy_trigger[buy_code] = "normal_replacement"
                        sell_pair_id[exit_code] = pair_id
                        buy_pair_id[buy_code] = pair_id
                        buy_gate_pair[buy_code] = pair_id

                for code in mandatory_exits:
                    sell_trigger.setdefault(
                        code,
                        "index_removal"
                        if not bool(execution_membership.get(code, False))
                        else "factor_missing_5d",
                    )

            # 单股上限优先于总敞口恢复，超限只减到1.5%。
            cap_notional = signal_close_nav * cfg.single_stock_weight_cap
            for code in held_codes:
                if code in mandatory_exits:
                    continue
                if signal_close_holdings[code] > cap_notional + 1e-8:
                    single_cap_breach_days[code] += 1
                    desired_close[code] = min(desired_close[code], cap_notional)
                    sell_trigger[code] = "single_name_cap"
                else:
                    single_cap_breach_days[code] = 0

            signal_stock_exposure = float(signal_close_holdings.sum()) / signal_close_nav
            exposure_band_breached = (
                signal_stock_exposure < cfg.stock_exposure_lower
                or signal_stock_exposure > cfg.stock_exposure_upper
            )
            if exposure_band_breached and desired_close.gt(1e-8).any():
                target_total = signal_close_nav * cfg.target_stock_exposure
                scale = target_total / float(desired_close.sum())
                desired_close = desired_close * scale
                desired_close = desired_close.clip(upper=cap_notional)
                for code in desired_close.index[desired_close.gt(signal_close_holdings + 1e-8)]:
                    buy_trigger.setdefault(code, "total_exposure_band")
                for code in desired_close.index[desired_close.lt(signal_close_holdings - 1e-8)]:
                    sell_trigger.setdefault(code, "total_exposure_band")

            # 将信号日收盘目标按持仓比例映射到执行日开盘名义金额。
            desired[:] = 0.0
            for code in factor.columns:
                close_value = float(signal_close_holdings[code])
                target_close_value = float(desired_close[code])
                if close_value > 1e-8 and holdings[code] > 1e-8:
                    desired[code] = holdings[code] * target_close_value / close_value
                elif target_close_value > 1e-8:
                    desired[code] = target_close_value

            signal_available = signal_is_valid or bool(mandatory_exits)
            rebalance = (
                day_number < len(dates) - 1
                and bool(
                    (desired - holdings).abs().gt(1e-8).any()
                )
            )
            top_codes = entry_codes
        else:
            top_codes, signal_valid_count = select_top_group(
                adjusted_factor.loc[factor_signal_date],
                factor_membership,
                min_stocks=cfg.min_stocks,
                quantile=cfg.quantile,
            )
            signal_available = bool(top_codes)
            rebalance = scheduled and signal_available and day_number < len(dates) - 1
            if rebalance:
                desired[:] = 0.0
                per_stock = nav_pretrade * cfg.target_stock_exposure / len(top_codes)
                desired.loc[top_codes] = per_stock

        # Exit-only exception: keep this stock unchanged at the open; liquidate
        # at the scheduled close. No intraday sale proceeds fund morning buys.
        if cfg.idealized_constituent_exit:
            for code in pending_index_removals:
                desired[code] = holdings[code]

        stock_adv = stock_adv_panel.loc[execution_info_date]
        market_cap_avg = market_cap_avg_panel.loc[execution_info_date]
        can_trade = day_raw_open.notna() & ~full_day_suspension.loc[date].fillna(False)

        stock_commission = stock_stamp = stock_transfer = stock_other = 0.0
        stock_slippage = stock_impact_cost = 0.0
        stock_requested = stock_filled = 0.0
        blocked_stock_orders = 0
        stock_sell_proceeds = 0.0
        stock_buy_cash = 0.0
        stock_bought_today = pd.Series(0.0, index=factor.columns, dtype=float)
        stock_sold_today = pd.Series(0.0, index=factor.columns, dtype=float)
        completed_sell_pairs: set[str] = set()
        for posting in pending_exit_postings:
            stock_commission += posting["commission"]
            stock_stamp += posting["stamp_tax"]
            stock_transfer += posting["transfer_fee"]
            stock_other += posting["other_fee"]
            stock_slippage += posting["slippage_cost"]
            stock_impact_cost += posting["impact_cost"]
            stock_sell_proceeds += posting["filled_notional"]
            stock_requested += posting["filled_notional"]
            stock_filled += posting["filled_notional"]
        posted_exit_count = len(pending_exit_postings)
        pending_exit_postings = []

        # 先卖出或减仓。
        sell_requests = (holdings - desired).clip(lower=0.0) if rebalance else holdings * 0.0
        sell_order_codes = sell_requests.index[sell_requests.gt(1e-8)].tolist()
        if independent_stock_sets:
            # Keep forced/risk priority; ordinary exits remain worst-Long-first.
            sell_order_codes.sort(key=lambda c: (sell_trigger.get(c)=="normal_replacement",
                                                 -int(exact_ranks.get(c, 10**9)), c))
        for code in sell_order_codes:
            requested = float(sell_requests[code])
            fill, execution_price, reason, impact_bp = calculate_stock_fill(
                requested_notional=requested,
                side="sell",
                open_price=float(day_raw_open[code]),
                up_limit=float(day_up_limit[code]),
                down_limit=float(day_down_limit[code]),
                adv_notional=float(stock_adv[code]),
                current_notional=float(holdings[code]),
                max_holding_notional=None,
                can_trade=bool(can_trade[code]),
                config=cfg,
                price_field=execution_price_field,
            )
            order_rows.append({
                "date": date, "signal_date": factor_signal_date,
                "execution_info_date": execution_info_date, "asset_type": "stock",
                "code": code, "side": "sell", "requested_notional": requested,
                "filled_notional": fill, "fill_ratio": fill / requested,
                "unfilled_notional": requested - fill, "execution_price": execution_price,
                "unfilled_reason": reason if fill < requested - 1e-8 else "",
                "adv_notional": float(stock_adv[code]), "impact_bp": impact_bp,
                "trigger_reason": sell_trigger.get(code, "full_rebalance"),
                "replacement_pair_id": sell_pair_id.get(code, ""),
                "paired_buy_cancelled_due_to_failed_sell": False,
            })
            stock_requested += requested
            stock_filled += fill
            if fill <= 0:
                blocked_stock_orders += 1
                continue
            fees = calculate_stock_fees(fill, "sell", date, cfg)
            total_bp = cfg.stock_fixed_slippage_bp + impact_bp
            slip = round_money(fill * total_bp / 10_000)
            impact_cost = round_money(fill * impact_bp / 10_000)
            cash += fill - slip - fees["total_transaction_cost"]
            holdings[code] -= fill
            stock_sold_today[code] += fill
            if holdings[code] <= 1e-8:
                holdings[code] = 0.0
                last_qfq[code] = np.nan
                missing_factor_days[code] = 0
                single_cap_breach_days[code] = 0
            pair_id = sell_pair_id.get(code)
            if pair_id and fill >= requested - 1e-8 and holdings[code] <= 1e-8:
                completed_sell_pairs.add(pair_id)
            stock_sell_proceeds += fill
            stock_commission += fees["commission"]
            stock_stamp += fees["stamp_tax"]
            stock_transfer += fees["transfer_fee"]
            stock_other += fees["other_fee"]
            stock_slippage += slip
            stock_impact_cost += impact_cost
            trade_rows.append({
                "date": date, "asset_type": "stock", "code": code, "side": "sell",
                "filled_notional": fill, "execution_price": execution_price,
                "commission": fees["commission"], "stamp_tax": fees["stamp_tax"],
                "transfer_fee": fees["transfer_fee"], "other_fee": fees["other_fee"],
                "transaction_cost": fees["total_transaction_cost"], "slippage_cost": slip,
                "impact_cost": impact_cost,
                "trigger_reason": sell_trigger.get(code, "full_rebalance"),
                "replacement_pair_id": sell_pair_id.get(code, ""),
            })

        # 生成候选买单，再根据现金和预计保证金统一按比例缩减。
        buy_candidates: dict[str, tuple[float, float, str, float]] = {}
        buy_requests = (desired - holdings).clip(lower=0.0) if rebalance else holdings * 0.0
        buy_order_codes = buy_requests.index[buy_requests.gt(1e-8)].tolist()
        if independent_stock_sets:
            priority={code:i for i,code in enumerate(independent_buy_order)}
            buy_order_codes.sort(key=lambda c: (holdings[c]<=1e-8, priority.get(c,10**9),c))
        reserved_slots = 0
        free_slots = max(0,cfg.target_stock_count-int(holdings.gt(1e-8).sum()))
        for code in buy_order_codes:
            requested = float(buy_requests[code])
            if independent_stock_sets and holdings[code]<=1e-8 and reserved_slots>=free_slots:
                buy_candidates[code]=(0.0,np.nan,"no_free_stock_slot",0.0)
                continue
            gate_pair = buy_gate_pair.get(code)
            if gate_pair and gate_pair not in completed_sell_pairs:
                buy_candidates[code] = (0.0, np.nan, "paired_sell_not_completed", 0.0)
                continue
            buy_candidates[code] = calculate_stock_fill(
                requested_notional=requested,
                side="buy",
                open_price=float(day_raw_open[code]),
                up_limit=float(day_up_limit[code]),
                down_limit=float(day_down_limit[code]),
                adv_notional=float(stock_adv[code]),
                current_notional=float(holdings[code]),
                max_holding_notional=(
                    float(market_cap_avg[code]) * cfg.circulating_market_cap_rate
                    if np.isfinite(market_cap_avg[code])
                    else None
                ),
                can_trade=bool(can_trade[code]),
                config=cfg,
                price_field=execution_price_field,
            )
            if independent_stock_sets and holdings[code]<=1e-8 and buy_candidates[code][0]>1e-8:
                reserved_slots += 1

        target_contract = target_contract_for(date) if futures_hedge_enabled else None
        estimated_margin = 0.0
        if target_contract is not None:
            target_row = future_row(target_contract, date)
            multiplier = float(futures_basic_lookup.at[target_contract, "multiplier"])
            margin_raw = future_parameter(target_contract, date, "short_margin_rate")
            if not np.isfinite(margin_raw):
                raise RuntimeError(f"{date.date()} {target_contract}缺少历史保证金率")
            projected_stock = float(holdings.sum()) + sum(v[0] for v in buy_candidates.values())
            projected_contracts = round_target_contracts(
                projected_stock
                / (float(target_row[execution_price_field]) * multiplier),
                cfg.futures_contract_rounding,
            )
            estimated_margin = (
                projected_contracts
                * float(target_row[execution_price_field])
                * multiplier
                * (margin_raw / 100 + cfg.broker_margin_addon)
            )

        def buy_cash_required(scale: float) -> float:
            total = 0.0
            for fill, _, _, impact_bp in buy_candidates.values():
                scaled = fill * scale
                if scaled <= 1e-8:
                    continue
                fees = calculate_stock_fees(scaled, "buy", date, cfg)
                total += scaled + fees["total_transaction_cost"]
                total += round_money(
                    scaled * (cfg.stock_fixed_slippage_bp + impact_bp * sqrt(scale)) / 10_000
                )
            return total

        cash_budget = max(0.0, cash - estimated_margin)
        buy_scale = 1.0
        if buy_cash_required(1.0) > cash_budget:
            low, high = 0.0, 1.0
            for _ in range(50):
                middle = (low + high) / 2
                if buy_cash_required(middle) <= cash_budget:
                    low = middle
                else:
                    high = middle
            buy_scale = low

        for code in buy_order_codes:
            requested = float(buy_requests[code])
            base_fill, _, reason, base_impact_bp = buy_candidates[code]
            fill = base_fill * buy_scale
            impact_bp = base_impact_bp * sqrt(buy_scale) if buy_scale > 0 else 0.0
            execution_price = np.nan
            if fill > 1e-8:
                total_bp = cfg.stock_fixed_slippage_bp + impact_bp
                execution_price = float(day_raw_open[code]) * (1 + total_bp / 10_000)
                if buy_scale < 1 - 1e-10:
                    reason = "cash_and_margin_capacity"
            order_rows.append({
                "date": date, "signal_date": factor_signal_date,
                "execution_info_date": execution_info_date, "asset_type": "stock",
                "code": code, "side": "buy", "requested_notional": requested,
                "filled_notional": fill, "fill_ratio": fill / requested,
                "unfilled_notional": requested - fill, "execution_price": execution_price,
                "unfilled_reason": reason if fill < requested - 1e-8 else "",
                "adv_notional": float(stock_adv[code]), "impact_bp": impact_bp,
                "trigger_reason": buy_trigger.get(code, "full_rebalance"),
                "replacement_pair_id": buy_pair_id.get(code, ""),
                "paired_buy_cancelled_due_to_failed_sell": (
                    reason == "paired_sell_not_completed"
                ),
            })
            stock_requested += requested
            stock_filled += fill
            if fill <= 1e-8:
                blocked_stock_orders += 1
                continue
            fees = calculate_stock_fees(fill, "buy", date, cfg)
            slip = round_money(fill * (cfg.stock_fixed_slippage_bp + impact_bp) / 10_000)
            impact_cost = round_money(fill * impact_bp / 10_000)
            cash -= fill + slip + fees["total_transaction_cost"]
            holdings[code] += fill
            stock_bought_today[code] += fill
            last_qfq[code] = float(day_qfq_open[code])
            stock_buy_cash += fill
            stock_commission += fees["commission"]
            stock_transfer += fees["transfer_fee"]
            stock_other += fees["other_fee"]
            stock_slippage += slip
            stock_impact_cost += impact_cost
            trade_rows.append({
                "date": date, "asset_type": "stock", "code": code, "side": "buy",
                "filled_notional": fill, "execution_price": execution_price,
                "commission": fees["commission"], "stamp_tax": fees["stamp_tax"],
                "transfer_fee": fees["transfer_fee"], "other_fee": fees["other_fee"],
                "transaction_cost": fees["total_transaction_cost"], "slippage_cost": slip,
                "impact_cost": impact_cost,
                "trigger_reason": buy_trigger.get(code, "full_rebalance"),
                "replacement_pair_id": buy_pair_id.get(code, ""),
            })

        stock_fee_total = round_money(
            stock_commission + stock_stamp + stock_transfer + stock_other
        )

        # IF开盘交易：旧仓先换月/减仓，再按实际股票市值补足对冲。
        starting_positions = futures_positions.copy()
        remaining_start_contracts = {
            contract: abs(position) for contract, position in starting_positions.items()
        }
        opened_lots: dict[str, list[tuple[int, float]]] = {}
        used_capacity: dict[str, int] = {}
        futures_fee = futures_submission = futures_slippage_cost = 0.0
        futures_requested = futures_filled = 0
        roll_deferred = False
        roll_unhedged = False
        delivery_proxy_count = 0

        def multiplier_for(contract: str) -> float:
            return float(futures_basic_lookup.at[contract, "multiplier"])

        def execute_future(contract: str, side: str, requested: int, purpose: str) -> int:
            nonlocal cash, futures_fee, futures_submission, futures_slippage_cost
            nonlocal futures_requested, futures_filled
            if requested <= 0:
                return 0
            row = future_row(contract, date)
            adv_value, adv_count = future_adv(contract, execution_info_date)
            fill, execution_price, impact_bp, reason = _futures_trade_capacity(
                requested_contracts=requested,
                side=side,
                row=row,
                futures_adv=adv_value,
                used_capacity=used_capacity.get(contract, 0),
                config=cfg,
                price_field=execution_price_field,
            )
            futures_requested += requested
            futures_filled += fill
            historical_close = (
                min(fill, remaining_start_contracts.get(contract, 0))
                if side == "buy"
                else 0
            )
            close_today = fill - historical_close if side == "buy" else 0
            submitted = int(fill > 0)
            cancelled = int(fill > 0 and fill < requested)
            submission_fee = (submitted + cancelled) * cfg.futures_submission_fee
            futures_submission += submission_fee
            cash -= submission_fee
            order_rows.append({
                "date": date, "signal_date": factor_signal_date,
                "execution_info_date": execution_info_date, "asset_type": "future",
                "code": contract, "side": side, "purpose": purpose,
                "requested_contracts": requested, "filled_contracts": fill,
                "unfilled_contracts": requested - fill,
                "fill_ratio": fill / requested, "execution_price": execution_price,
                "unfilled_reason": reason if fill < requested else "",
                "adv_observation_count": adv_count, "adv_full_window": adv_count >= cfg.futures_adv_window,
                "futures_adv": adv_value, "impact_bp": impact_bp,
                "historical_close_contracts": historical_close,
                "close_today_contracts": close_today,
                "submitted_order_count": submitted, "cancelled_order_count": cancelled,
                "submission_fee": submission_fee,
            })
            if fill <= 0:
                return 0
            used_capacity[contract] = used_capacity.get(contract, 0) + fill
            mult = multiplier_for(contract)
            notional_per_contract = execution_price * mult
            if side == "buy" and close_today > 0:
                offset_multiplier = future_parameter(
                    contract,
                    date,
                    "offset_today_fee",
                )
                if not np.isfinite(offset_multiplier):
                    raise RuntimeError(f"{date.date()} {contract}缺少历史平今费率")
                close_today_rate = (
                    cfg.futures_exchange_fee_rate * offset_multiplier / 100
                )
            else:
                close_today_rate = cfg.futures_exchange_fee_rate
            exchange_fee = notional_per_contract * (
                (fill - close_today) * cfg.futures_exchange_fee_rate
                + close_today * close_today_rate
            )
            fee = round_money(exchange_fee * cfg.futures_broker_fee_multiplier)
            futures_fee += fee
            cash -= fee
            open_price = float(row[execution_price_field])
            futures_slippage_cost += abs(execution_price - open_price) * mult * fill
            if side == "buy":
                available = abs(futures_positions.get(contract, 0))
                close_count = min(fill, available)
                if historical_close > 0:
                    basis = futures_basis[contract]
                    cash += (-historical_close) * mult * (execution_price - basis)
                remaining_start_contracts[contract] = max(
                    0,
                    remaining_start_contracts.get(contract, 0) - historical_close,
                )
                remaining_today = close_today
                updated_lots: list[tuple[int, float]] = []
                for lot_qty, lot_price in opened_lots.get(contract, []):
                    lot_size = abs(lot_qty)
                    lot_close = min(lot_size, remaining_today)
                    if lot_close:
                        cash += (-lot_close) * mult * (execution_price - lot_price)
                        remaining_today -= lot_close
                    residual = lot_size - lot_close
                    if residual:
                        updated_lots.append((-residual, lot_price))
                opened_lots[contract] = updated_lots
                if remaining_today:
                    raise RuntimeError(
                        f"{date.date()} {contract}平今手数超过当日新开仓手数"
                    )
                futures_positions[contract] = futures_positions.get(contract, 0) + close_count
                if futures_positions[contract] == 0:
                    futures_positions.pop(contract, None)
                    futures_basis.pop(contract, None)
            else:
                futures_positions[contract] = futures_positions.get(contract, 0) - fill
                opened_lots.setdefault(contract, []).append((-fill, execution_price))
            trade_rows.append({
                "date": date, "asset_type": "future", "code": contract,
                "side": side, "purpose": purpose, "filled_contracts": fill,
                "historical_close_contracts": historical_close,
                "close_today_contracts": close_today,
                "execution_price": execution_price, "transaction_cost": fee,
                "submission_fee": submission_fee,
                "slippage_cost": abs(execution_price - open_price) * mult * fill,
            })
            return fill

        # 保证金不足时优先减少旧空头。
        if margin_call_pending:
            for contract in sorted(list(futures_positions)):
                execute_future(contract, "buy", abs(futures_positions[contract]), "margin_call")

        # 在到期前按相同手数配对换月。
        if target_contract is not None:
            for old_contract in sorted(list(futures_positions)):
                if old_contract == target_contract:
                    continue
                old_info = futures_basic_lookup.loc[old_contract]
                old_qty = abs(futures_positions[old_contract])
                old_row = future_row(old_contract, date)
                new_row = future_row(target_contract, date)
                old_adv, _ = future_adv(old_contract, execution_info_date)
                new_adv, _ = future_adv(target_contract, execution_info_date)
                old_cap = max(0, floor(old_adv * cfg.futures_participation_rate)) if np.isfinite(old_adv) else 0
                new_cap = max(0, floor(new_adv * cfg.futures_participation_rate)) if np.isfinite(new_adv) else 0
                pair_request = min(old_qty, old_cap, new_cap)
                if date < pd.Timestamp(old_info["delist_date"]):
                    old_test = _futures_trade_capacity(
                        requested_contracts=pair_request, side="buy", row=old_row,
                        futures_adv=old_adv, used_capacity=used_capacity.get(old_contract, 0), config=cfg,
                        price_field=execution_price_field,
                    )
                    new_test = _futures_trade_capacity(
                        requested_contracts=pair_request, side="sell", row=new_row,
                        futures_adv=new_adv, used_capacity=used_capacity.get(target_contract, 0), config=cfg,
                        price_field=execution_price_field,
                    )
                    paired = min(old_test[0], new_test[0])
                    if paired > 0:
                        closed = execute_future(old_contract, "buy", paired, "roll_close")
                        opened = execute_future(target_contract, "sell", closed, "roll_open")
                        roll_rows.append({
                            "date": date, "old_contract": old_contract,
                            "new_contract": target_contract, "requested_contracts": old_qty,
                            "paired_roll_contracts": min(closed, opened),
                            "roll_deferred": min(closed, opened) < old_qty,
                            "roll_unhedged": False,
                        })
                    if paired < old_qty:
                        roll_deferred = True
                else:
                    closed = execute_future(old_contract, "buy", old_qty, "expiry_close")
                    if closed < old_qty:
                        settle_row = future_row(old_contract, date)
                        settle_price = np.nan if settle_row is None else float(settle_row["settle"])
                        if not np.isfinite(settle_price):
                            raise RuntimeError(f"{date.date()} {old_contract}缺少到期结算代理价格")
                        remaining = abs(futures_positions.get(old_contract, 0))
                        mult = multiplier_for(old_contract)
                        cash += (-remaining) * mult * (settle_price - futures_basis[old_contract])
                        futures_positions.pop(old_contract, None)
                        futures_basis.pop(old_contract, None)
                        delivery_proxy_count += int(remaining > 0)
                    roll_unhedged = True

        # 根据实际股票名义市值调整总空头手数。
        desired_contracts = 0
        raw_target_contracts = 0.0
        if target_contract is not None:
            target_row = future_row(target_contract, date)
            target_multiplier = multiplier_for(target_contract)
            raw_target_contracts = (
                float(holdings.sum())
                / (float(target_row[execution_price_field]) * target_multiplier)
            )
            desired_contracts = round_target_contracts(
                raw_target_contracts,
                cfg.futures_contract_rounding,
            )
            if margin_call_pending:
                margin_raw = future_parameter(target_contract, date, "short_margin_rate")
                affordable = floor(
                    max(0.0, cash)
                    / (
                        float(target_row[execution_price_field])
                        * target_multiplier
                        * (margin_raw / 100 + cfg.broker_margin_addon)
                    )
                )
                desired_contracts = min(desired_contracts, affordable)
            current_contracts = sum(abs(v) for v in futures_positions.values())
            if current_contracts < desired_contracts:
                execute_future(
                    target_contract,
                    "sell",
                    desired_contracts - current_contracts,
                    "hedge_increase",
                )
            elif current_contracts > desired_contracts:
                remaining_to_close = current_contracts - desired_contracts
                for contract in sorted(
                    list(futures_positions),
                    key=lambda item: (item == target_contract, item),
                ):
                    if remaining_to_close <= 0:
                        break
                    filled = execute_future(
                        contract,
                        "buy",
                        min(remaining_to_close, abs(futures_positions[contract])),
                        "hedge_decrease",
                    )
                    remaining_to_close -= filled

        # 当前开盘成交后的期货未实现价值。
        futures_open_value_after = 0.0
        for contract, position in futures_positions.items():
            row = future_row(contract, date)
            if row is None or not np.isfinite(row[execution_price_field]):
                continue
            mult = multiplier_for(contract)
            starting_remaining = remaining_start_contracts.get(contract, 0)
            if starting_remaining and contract in futures_basis:
                futures_open_value_after += (-starting_remaining) * mult * (
                    float(row[execution_price_field]) - futures_basis[contract]
                )
            for lot_qty, lot_price in opened_lots.get(contract, []):
                futures_open_value_after += lot_qty * mult * (
                    float(row[execution_price_field]) - lot_price
                )

        nav_after_open = cash + float(holdings.sum()) + futures_open_value_after
        cash_after_open = cash
        total_cost = stock_fee_total + stock_slippage + futures_fee + futures_submission + futures_slippage_cost
        net_return = nav_after_open / previous_nav - 1.0
        portfolio_pnl = nav_after_open - previous_nav
        futures_period_pnl = portfolio_pnl - stock_market_pnl + total_cost
        if not futures_hedge_enabled:
            if abs(futures_period_pnl)>1e-5:
                raise AssertionError("Stock-only cash/holdings do not reconcile")
            futures_period_pnl=0.0
        gross_market_pnl = stock_market_pnl + futures_period_pnl
        gross_return = gross_market_pnl / previous_nav

        if cash < -0.005:
            snapshot = {
                "date": date,
                "factor_name": factor_name,
                "cash": cash,
                "nav_after_open": nav_after_open,
                "stock_holdings": holdings[holdings.gt(0)].to_dict(),
                "futures_positions": futures_positions.copy(),
                "last_orders": order_rows[-20:],
                "last_trades": trade_rows[-20:],
            }
            raise NegativeCashError(f"{date.date()}开盘后现金为负", snapshot)

        stock_value = float(holdings.sum())
        futures_notional = 0.0
        for contract, position in futures_positions.items():
            row = future_row(contract, date)
            if row is not None and np.isfinite(row[execution_price_field]):
                futures_notional += (
                    abs(position)
                    * float(row[execution_price_field])
                    * multiplier_for(contract)
                )

        # 同行保存目标、漂移后成交前和成交后三种状态，不能用目标冒充成交。
        audited_codes = factor.columns[
            desired.gt(1e-8) | pretrade_holdings.gt(1e-8) | holdings.gt(1e-8)
        ]
        for code in audited_codes:
            sellable_after_open = min(
                float(holdings[code]),
                max(0.0, float(pretrade_holdings[code] - stock_sold_today[code])),
            )
            holding_rows.append({
                "date": date, "signal_date": factor_signal_date,
                "execution_info_date": execution_info_date, "code": code,
                "target_notional": float(desired[code]),
                "pretrade_notional": float(pretrade_holdings[code]),
                "posttrade_notional": float(holdings[code]),
                "target_weight": float(desired[code] / nav_pretrade),
                "pretrade_weight": float(pretrade_holdings[code] / nav_pretrade),
                "posttrade_weight": float(holdings[code] / nav_after_open),
                "holding_period_market_pnl": float(stock_pnl_by_code[code]),
                "bought_today_notional": float(stock_bought_today[code]),
                "sold_today_notional": float(stock_sold_today[code]),
                "sellable_notional": sellable_after_open,
                "t_plus_one_blocked_notional": float(holdings[code] - sellable_after_open),
                "stale_price_days": int(stale_days[code]),
                "is_member_on_signal_date": bool(factor_membership[code]),
                "is_member_on_execution_date": bool(execution_membership[code]),
                "signal_close_weight": (
                    float(signal_close_holdings[code] / signal_close_nav)
                    if cfg.strategy_mode == "buffered_v2" and signal_close_nav > 0
                    else np.nan
                ),
                "exact_factor_rank": (
                    int(exact_ranks[code]) if code in exact_ranks.index else pd.NA
                ),
                "in_top15_entry_set": code in set(entry_codes),
                "in_top25_retention_set": code in set(retention_codes),
                "consecutive_factor_missing_days": int(missing_factor_days[code]),
                "single_stock_cap_excess": (
                    max(
                        0.0,
                        float(
                            signal_close_holdings[code]
                            - signal_close_nav * cfg.single_stock_weight_cap
                        ),
                    )
                    if cfg.strategy_mode == "buffered_v2"
                    else 0.0
                ),
                "single_stock_cap_breach_days": int(single_cap_breach_days[code]),
            })
        for contract, position in futures_positions.items():
            position_row = future_row(contract, date)
            position_open = (
                float(position_row[execution_price_field])
                if position_row is not None
                and np.isfinite(position_row[execution_price_field])
                else np.nan
            )
            futures_position_rows.append({
                "date": date, "contract": contract, "contracts": position,
                "open_price": position_open,
                "valuation_price_field": execution_price_field,
                "notional": (
                    abs(position) * position_open * multiplier_for(contract)
                    if np.isfinite(position_open)
                    else np.nan
                ),
            })

        # 收盘按结算价盯市并重置下一日成本基础。
        settlement_pnl = 0.0
        required_margin = 0.0
        for contract, position in list(futures_positions.items()):
            row = future_row(contract, date)
            if row is None or not np.isfinite(row["settle"]):
                raise RuntimeError(f"{date.date()} {contract}缺少结算价")
            settle_price = float(row["settle"])
            mult = multiplier_for(contract)
            starting_remaining = remaining_start_contracts.get(contract, 0)
            if starting_remaining and contract in futures_basis:
                settlement_pnl += (-starting_remaining) * mult * (
                    settle_price - futures_basis[contract]
                )
            for lot_qty, lot_price in opened_lots.get(contract, []):
                settlement_pnl += lot_qty * mult * (settle_price - lot_price)
            futures_basis[contract] = settle_price
            margin_raw = future_parameter(contract, date, "short_margin_rate")
            if not np.isfinite(margin_raw):
                raise RuntimeError(f"{date.date()} {contract}缺少历史保证金率")
            required_margin += abs(position) * settle_price * mult * (
                margin_raw / 100 + cfg.broker_margin_addon
            )
        cash += settlement_pnl
        margin_call_pending = cash < required_margin
        if cash < -0.005:
            snapshot = {
                "date": date,
                "factor_name": factor_name,
                "cash": cash,
                "required_margin": required_margin,
                "settlement_pnl": settlement_pnl,
                "stock_holdings": holdings[holdings.gt(0)].to_dict(),
                "futures_positions": futures_positions.copy(),
            }
            raise NegativeCashError(f"{date.date()}期货结算后现金为负", snapshot)

        daily_rows.append({
            "date": date,
            "signal_date": factor_signal_date,
            "execution_info_date": execution_info_date,
            "factor_signal_lag": factor_signal_lag,
            "execution_price_field": execution_price_field,
            **veto_diagnostics,
            "is_scheduled_rebalance": (
                scheduled
            ),
            "rebalance_executed": rebalance,
            "signal_valid_count": signal_valid_count,
            "target_stock_count": (
                cfg.target_stock_count
                if cfg.strategy_mode == "buffered_v2"
                else (len(top_codes) if scheduled else 0)
            ),
            "actual_stock_count": int(holdings.gt(1e-8).sum()),
            "target_slots": (
                cfg.target_stock_count if cfg.strategy_mode == "buffered_v2" else len(top_codes)
            ),
            "occupied_slots": int(holdings.gt(1e-8).sum()),
            "vacant_slots": (
                max(0, cfg.target_stock_count - int(holdings.gt(1e-8).sum()))
                if cfg.strategy_mode == "buffered_v2"
                else 0
            ),
            "entry_set_count": len(entry_codes),
            "retention_set_count": len(retention_codes),
            "initial_construction": initial_construction,
            "stock_exposure_band_breached": exposure_band_breached,
            "signal_close_stock_exposure": (
                float(signal_close_holdings.sum() / signal_close_nav)
                if cfg.strategy_mode == "buffered_v2" and signal_close_nav > 0
                else np.nan
            ),
            "max_consecutive_factor_missing_days": int(missing_factor_days.max()),
            "normal_exit_candidates_before_ndrop": normal_exit_candidates_before_ndrop,
            "normal_replacement_requests_before_ndrop": (
                normal_replacement_requests_before_ndrop
            ),
            "normal_replacements_blocked_by_ndrop": (
                normal_replacements_blocked_by_ndrop
            ),
            "cash_at_start": cash_at_start,
            "cash_after_open": cash_after_open,
            "nav_pretrade": nav_pretrade,
            "nav_after_open": nav_after_open,
            "net_return": net_return,
            "gross_return_before_cost": gross_return,
            "portfolio_pnl": portfolio_pnl,
            "gross_market_pnl": gross_market_pnl,
            "stock_market_pnl": stock_market_pnl,
            "futures_hedge_pnl": futures_period_pnl,
            "futures_open_value_before": futures_open_value_before,
            "futures_open_value_after": futures_open_value_after,
            "stock_value": stock_value,
            "stock_exposure": stock_value / nav_after_open,
            "futures_short_notional": futures_notional,
            "futures_short_exposure": futures_notional / nav_after_open,
            "gross_exposure": (stock_value + futures_notional) / nav_after_open,
            "net_exposure": (stock_value - futures_notional) / nav_after_open,
            "cash_balance": cash,
            "cash_after_settlement": cash,
            "required_margin": required_margin,
            "available_cash_after_margin": cash - required_margin,
            "margin_call": margin_call_pending,
            "target_futures_contract": target_contract,
            "raw_target_if_contracts": raw_target_contracts,
            "target_short_contracts": desired_contracts,
            "actual_short_contracts": sum(abs(v) for v in futures_positions.values()),
            "hedge_unfilled_notional": max(0.0, stock_value - futures_notional) if futures_hedge_enabled else 0.0,
            "hedge_rounding_notional_error": stock_value - futures_notional if futures_hedge_enabled else 0.0,
            "stock_requested_notional": stock_requested,
            "stock_filled_notional": stock_filled,
            "stock_fill_ratio": stock_filled / stock_requested if stock_requested else np.nan,
            "blocked_stock_orders": blocked_stock_orders,
            "stock_buy_notional": stock_buy_cash,
            "stock_sell_notional": stock_sell_proceeds,
            "stock_traded_notional": stock_buy_cash + stock_sell_proceeds,
            "new_stock_positions": int(
                (stock_bought_today.gt(1e-8) & pretrade_holdings.le(1e-8)).sum()
            ),
            "closed_stock_positions": int(
                (stock_sold_today.gt(1e-8) & holdings.le(1e-8)).sum()
            ),
            "stock_turnover": (
                (stock_buy_cash + stock_sell_proceeds) / nav_pretrade
                if nav_pretrade > 0
                else np.nan
            ),
            "stock_commission": stock_commission,
            "stamp_tax": stock_stamp,
            "transfer_fee": stock_transfer,
            "other_fee": stock_other,
            "stock_transaction_cost": stock_fee_total,
            "stock_slippage_cost": stock_slippage,
            "stock_impact_cost": stock_impact_cost,
            "futures_requested_contracts": futures_requested,
            "futures_filled_contracts": futures_filled,
            "futures_transaction_cost": futures_fee,
            "futures_submission_fee": futures_submission,
            "futures_slippage_cost": futures_slippage_cost,
            "futures_settlement_pnl": settlement_pnl,
            "total_cost": total_cost,
            "roll_deferred": roll_deferred,
            "roll_unhedged": roll_unhedged,
            "delivery_settlement_proxy_count": delivery_proxy_count,
            "max_stale_price_days": int(stale_days[holdings.gt(0)].max()) if holdings.gt(0).any() else 0,
        })
        # Authorized constituent exits happen only after the day's open book
        # and futures settlement. The stock ledger posts at the next open.
        close_exits_today = []
        if cfg.idealized_constituent_exit and scheduled:
            for code in sorted(pending_index_removals):
                before = float(holdings[code])
                if before <= 1e-8:
                    continue
                valuation = value_exit(
                    holding_notional=before, mark_basis=float(last_qfq[code]),
                    current_close=float(qfq_close.at[date, code]),
                    latest_close=float(close_asof.at[date, code]),
                    raw_close=float(raw_close_asof.at[date, code]),
                    prior_adv=float(stock_adv[code]),
                    impact_coefficient=cfg.stock_impact_coefficient_bp,
                )
                fill = valuation.proceeds_before_cost
                fees = calculate_stock_fees(fill, "sell", date, cfg)
                slip = round_money(fill * (cfg.stock_fixed_slippage_bp + valuation.impact_bp) / 10_000)
                record = {
                    "date": date, "execution_session": "close", "asset_type": "stock",
                    "code": code, "side": "sell", "filled_notional": fill,
                    "requested_notional": fill, "unfilled_notional": 0.0, "fill_ratio": 1.0,
                    "execution_price": valuation.reference_price * (1-(cfg.stock_fixed_slippage_bp+valuation.impact_bp)/10000),
                    "reference_close": valuation.reference_price, "price_source": valuation.price_source,
                    "stale_close": valuation.stale_close,
                    "missing_adv_fixed_slippage_only": valuation.missing_adv_fixed_slippage_only,
                    "commission": fees["commission"], "stamp_tax": fees["stamp_tax"],
                    "transfer_fee": fees["transfer_fee"], "other_fee": fees["other_fee"],
                    "transaction_cost": fees["total_transaction_cost"], "slippage_cost": slip,
                    "impact_cost": round_money(fill*valuation.impact_bp/10000),
                    "market_pnl": valuation.market_pnl, "opening_notional_sold": before,
                    "trigger_reason": "index_removal_idealized_close", "idealized_exit": True,
                    "unfilled_reason": "", "posttrade_notional": 0.0,
                    "posting_date": dates[day_number+1] if day_number+1<len(dates) else date,
                    "terminal_same_day_posting": day_number==len(dates)-1,
                }
                cash += fill - slip - fees["total_transaction_cost"]
                holdings[code] = 0.0
                last_qfq[code] = np.nan
                missing_factor_days[code] = 0
                single_cap_breach_days[code] = 0
                trade_rows.append(record.copy()); order_rows.append(record.copy())
                exit_rows.append(record.copy()); close_exits_today.append(record)
            pending_index_removals.difference_update(r["code"] for r in close_exits_today)
            pending_exit_postings = close_exits_today
            margin_call_pending = cash < required_margin
        daily_rows[-1]["idealized_exit_count_executed_close"] = len(close_exits_today)
        daily_rows[-1]["idealized_exit_count_posted"] = posted_exit_count
        daily_rows[-1]["stock_value_after_exit_close"] = float(holdings.sum())
        daily_rows[-1]["exit_close_cash_after_settlement"] = cash
        # Preserve a final-date liquidation even though there is no next open.
        # Only removed stocks move to their close; other marks stay at the open.
        if close_exits_today and day_number == len(dates)-1:
            row = daily_rows[-1]
            pnl = sum(r["market_pnl"] for r in close_exits_today)
            cost = sum(r["transaction_cost"]+r["slippage_cost"] for r in close_exits_today)
            delta = pnl-cost
            row["terminal_exit_close_adjustment"] = delta
            row["nav_after_open"] += delta
            row["stock_value"] = float(holdings.sum())
            row["cash_after_open"] += sum(r["filled_notional"]-r["transaction_cost"]-r["slippage_cost"] for r in close_exits_today)
            row["stock_market_pnl"] += pnl
            row["gross_market_pnl"] += pnl
            row["portfolio_pnl"] += delta
            row["total_cost"] += cost
            for field, source_field in [("stock_transaction_cost","transaction_cost"),("stock_slippage_cost","slippage_cost"),("stock_impact_cost","impact_cost"),("stock_commission","commission"),("stamp_tax","stamp_tax"),("transfer_fee","transfer_fee"),("other_fee","other_fee"),("stock_sell_notional","filled_notional"),("stock_traded_notional","filled_notional")]:
                row[field] += sum(r[source_field] for r in close_exits_today)
            row["net_return"] = row["nav_after_open"] / previous_nav - 1
            row["gross_return_before_cost"] = row["gross_market_pnl"] / previous_nav
            row["stock_exposure"] = row["stock_value"] / row["nav_after_open"]
            row["futures_short_exposure"] = row["futures_short_notional"] / row["nav_after_open"]
            row["gross_exposure"] = (row["stock_value"]+row["futures_short_notional"]) / row["nav_after_open"]
            row["net_exposure"] = (row["stock_value"]-row["futures_short_notional"]) / row["nav_after_open"]
            row["stock_turnover"] = row["stock_traded_notional"] / row["nav_pretrade"]
            row["cash_balance"] = cash; row["cash_after_settlement"] = cash
            row["available_cash_after_margin"] = cash-required_margin
            row["idealized_exit_count_posted"] += len(close_exits_today)
            for hrow in holding_rows:
                if hrow["date"] == date and hrow["code"] in {r["code"] for r in close_exits_today}:
                    hrow["posttrade_notional"] = 0.0; hrow["posttrade_weight"] = 0.0
                if hrow["date"] == date:
                    hrow["posttrade_weight"] = hrow["posttrade_notional"] / row["nav_after_open"]
            pending_exit_postings = []
        previous_nav = daily_rows[-1]["nav_after_open"]
        previous_date = date

    daily = pd.DataFrame(daily_rows).set_index("date")
    daily["net_value"] = daily["nav_after_open"] / cfg.initial_capital
    daily["same_path_gross_net_value"] = (
        1.0 + daily["gross_return_before_cost"].fillna(0.0)
    ).cumprod()
    gross_metrics = _summary(
        daily["gross_return_before_cost"],
        daily["same_path_gross_net_value"],
    )
    net_metrics = _summary(daily["net_return"], daily["nav_after_open"])
    summary = pd.concat([
        pd.Series({
            "factor_name": factor_name,
            "backtest_status": "success",
            "selected_direction": float(selected_direction),
            "start_date": dates.min(),
            "end_date": dates.max(),
            "initial_capital": cfg.initial_capital,
            "rebalance_interval": cfg.rebalance_interval,
            "factor_signal_lag": factor_signal_lag,
            "execution_price_field": execution_price_field,
            "rebalance_frequency": "daily" if cfg.rebalance_interval == 1 else f"every_{cfg.rebalance_interval}_trading_days",
            "strategy_mode": cfg.strategy_mode,
            "target_stock_count": cfg.target_stock_count,
            "entry_fraction": cfg.entry_fraction,
            "retention_fraction": cfg.retention_fraction,
            "factor_missing_exit_days": cfg.factor_missing_exit_days,
            "max_signal_replacements_per_day": max_signal_replacements_per_day,
            "stock_exposure_lower": cfg.stock_exposure_lower,
            "stock_exposure_upper": cfg.stock_exposure_upper,
            "single_stock_weight_cap": cfg.single_stock_weight_cap,
            "futures_contract_rounding": cfg.futures_contract_rounding,
            "stock_impact_coefficient_bp": cfg.stock_impact_coefficient_bp,
            "futures_impact_coefficient_bp": cfg.futures_impact_coefficient_bp,
            "st_filter_applied": False,
            "historical_index_membership_applied": True,
            "transaction_cost_applied": True,
            "slippage_applied": True,
            "individual_stock_short_applied": False,
            "short_borrow_constraint_applied": False,
            "index_futures_hedge_applied": futures_hedge_enabled,
            "mean_stock_exposure": daily["stock_exposure"].mean(),
            "mean_futures_short_exposure": daily["futures_short_exposure"].mean(),
            "mean_net_exposure": daily["net_exposure"].mean(),
            "mean_cash_ratio": daily["cash_balance"].div(daily["nav_after_open"]).mean(),
            "mean_stock_turnover": daily["stock_turnover"].mean(),
            "mean_daily_new_stock_positions": daily["new_stock_positions"].mean(),
            "total_new_stock_positions": daily["new_stock_positions"].sum(),
            "total_stock_traded_notional": daily["stock_traded_notional"].sum(),
            "total_stock_transaction_cost": daily["stock_transaction_cost"].sum(),
            "total_stock_slippage_cost": daily["stock_slippage_cost"].sum(),
            "total_stock_cost": (
                daily["stock_transaction_cost"].sum()
                + daily["stock_slippage_cost"].sum()
            ),
            "total_futures_transaction_cost": daily["futures_transaction_cost"].sum(),
            "total_futures_submission_fee": daily["futures_submission_fee"].sum(),
            "total_futures_slippage_cost": daily["futures_slippage_cost"].sum(),
            "total_futures_cost": (
                daily["futures_transaction_cost"].sum()
                + daily["futures_submission_fee"].sum()
                + daily["futures_slippage_cost"].sum()
            ),
            "total_cost": daily["total_cost"].sum(),
            "total_blocked_stock_orders": daily["blocked_stock_orders"].sum(),
            "margin_call_days": daily["margin_call"].sum(),
            "roll_deferred_days": daily["roll_deferred"].sum(),
            "roll_unhedged_days": daily["roll_unhedged"].sum(),
            "delivery_settlement_proxy_count": daily["delivery_settlement_proxy_count"].sum(),
        }),
        net_metrics,
        pd.Series({f"gross_{key}": value for key, value in gross_metrics.items()}),
        pd.Series(
            {
                "gross_to_net_cumulative_return_drag": (
                    gross_metrics["cumulative_return"] - net_metrics["cumulative_return"]
                ),
                "total_cost_over_initial_nav": daily["total_cost"].sum()
                / cfg.initial_capital,
            }
        ),
    ])
    return {
        "summary": summary,
        "daily": daily,
        "yearly": _yearly_summary(daily),
        "holdings": pd.DataFrame(holding_rows),
        "futures_positions": pd.DataFrame(futures_position_rows),
        "orders": pd.DataFrame(order_rows),
        "trades": pd.DataFrame(trade_rows),
        "rolls": pd.DataFrame(roll_rows),
        "constituent_exits": pd.DataFrame(exit_rows),
        **({"veto_events": pd.DataFrame(veto_events)} if independent_stock_sets else {}),
    }
