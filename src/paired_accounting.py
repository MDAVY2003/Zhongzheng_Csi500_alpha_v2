"""Replay the hedged engine's stock ledger without altering any stock trades.

The replay is conditional on A's decisions, including its margin constraints.
It does not retarget stocks using B's NAV. Percentage weights can differ because
NAV differs; stock notionals, fees, and quantities represented by notionals agree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class InfeasibleSharedPath(ValueError):
    """The shared stock path cannot be funded without borrowing."""


def replay_stock_only(
    hedged: dict[str, object], *, initial_capital: float
) -> dict[str, pd.DataFrame]:
    """Remove futures PnL, fees, slippage, settlement, and margin from A.

    Open NAV = initial capital + cumulative stock market PnL - stock costs.
    Cash is reconciled both by NAV less holdings and by stock trade cashflows.
    Daily futures settlement must not be deducted again from open NAV.
    """
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError('initial_capital must be positive and finite')
    a = hedged['daily'].copy()
    required = {
        'stock_market_pnl', 'stock_transaction_cost', 'stock_slippage_cost',
        'stock_value', 'stock_buy_notional', 'stock_sell_notional',
        'nav_after_open', 'futures_hedge_pnl', 'futures_transaction_cost',
        'futures_submission_fee', 'futures_slippage_cost',
    }
    if missing := required.difference(a.columns):
        raise ValueError(f'Missing audited ledger fields: {sorted(missing)}')
    if not isinstance(a.index, pd.DatetimeIndex) or not a.index.is_unique or not a.index.is_monotonic_increasing or a.empty:
        raise ValueError('Ledger dates must be nonempty, unique, and increasing')
    if not np.isfinite(a[list(required)].to_numpy(dtype=float)).all():
        raise ValueError('Ledger contains missing or nonfinite values')
    nonnegative = ['stock_transaction_cost', 'stock_slippage_cost', 'stock_value',
                   'stock_buy_notional', 'stock_sell_notional',
                   'futures_transaction_cost', 'futures_submission_fee', 'futures_slippage_cost']
    if a[nonnegative].lt(0).any().any():
        raise ValueError('Notionals and costs must be nonnegative')
    stock_cost = a.stock_transaction_cost + a.stock_slippage_cost
    future_cost = a.futures_transaction_cost + a.futures_submission_fee + a.futures_slippage_cost
    nav = initial_capital + (a.stock_market_pnl - stock_cost).cumsum()
    removed_leg = (a.futures_hedge_pnl - future_cost).cumsum()
    if not np.allclose(a.nav_after_open - nav, removed_leg, atol=1e-5, rtol=1e-10):
        raise ValueError('A/B difference does not reconcile to the futures leg')
    cash = nav - a.stock_value
    cash_from_trades = initial_capital + (
        a.stock_sell_notional - a.stock_buy_notional - stock_cost
    ).cumsum()
    if not np.allclose(cash, cash_from_trades, atol=1e-5, rtol=1e-10):
        raise ValueError('Stock cashflows do not reconcile to stock PnL/holdings')
    if (cash < -0.005).any() or (nav <= 0).any():
        raise InfeasibleSharedPath('Shared stock path requires borrowing or has nonpositive NAV')
    previous_nav = nav.shift(1, fill_value=initial_capital)
    b = pd.DataFrame({
        'nav_after_open': nav,
        'net_value': nav / initial_capital,
        'net_return': (a.stock_market_pnl - stock_cost) / previous_nav,
        'gross_return_before_cost': a.stock_market_pnl / previous_nav,
        'stock_market_pnl': a.stock_market_pnl,
        'stock_value': a.stock_value,
        'stock_exposure': a.stock_value / nav,
        'cash_after_open': cash,
        'cash_interest': 0.0,
        'stock_transaction_cost': a.stock_transaction_cost,
        'stock_slippage_cost': a.stock_slippage_cost,
        'total_cost': stock_cost,
        'futures_hedge_pnl': 0.0,
        'futures_transaction_cost': 0.0,
        'futures_submission_fee': 0.0,
        'futures_slippage_cost': 0.0,
        'futures_short_notional': 0.0,
        'required_margin': 0.0,
        'stock_buy_notional': a.stock_buy_notional,
        'stock_sell_notional': a.stock_sell_notional,
        'removed_futures_net_pnl_cumulative': removed_leg,
    }, index=a.index)
    for column in ['idealized_exit_count_executed_close','idealized_exit_count_posted',
                   'stock_value_after_exit_close','terminal_exit_close_adjustment']:
        if column in a.columns:
            b[column] = a[column]
    outputs = {'daily': b}
    for key in ['holdings', 'orders', 'trades']:
        frame = hedged[key].copy(deep=True)
        if key != 'holdings':
            if 'asset_type' not in frame.columns:
                if not frame.empty:
                    raise ValueError(f'{key} has no asset_type')
            else:
                frame = frame.loc[frame.asset_type.eq('stock')].copy()
        else:
            # Keep the source weights labelled explicitly; recalculate B's weights.
            for column in ['pretrade_weight', 'posttrade_weight', 'signal_close_weight']:
                if column in frame.columns:
                    frame = frame.rename(columns={column: 'source_A_' + column})
            if {'date', 'posttrade_notional'}.issubset(frame.columns):
                frame['posttrade_weight'] = frame.posttrade_notional / frame.date.map(nav)
        outputs[key] = frame
    return outputs


def matched_benchmark(
    index_return: pd.Series, stock_exposure_at_previous_open: pd.Series,
    *, cash_return: pd.Series | None = None,
) -> pd.DataFrame:
    """Match B's exposure fixed before each open-to-open return interval.

    Inputs share the date of the interval's ending open. Caller must lag B's
    posttrade exposure by one trading day, including any pre-period initial state.
    Full cash earns the passed cash return, which is zero in the inherited model.
    """
    if not index_return.index.equals(stock_exposure_at_previous_open.index):
        raise ValueError('Index and pre-interval exposure dates differ')
    if not isinstance(index_return.index, pd.DatetimeIndex) or not index_return.index.is_unique or not index_return.index.is_monotonic_increasing:
        raise ValueError('Benchmark dates must be unique and increasing')
    cash = pd.Series(0.0, index=index_return.index) if cash_return is None else cash_return
    if not cash.index.equals(index_return.index):
        raise ValueError('Cash return dates differ')
    inputs = pd.concat([index_return, stock_exposure_at_previous_open, cash], axis=1)
    if not np.isfinite(inputs.to_numpy(dtype=float)).all():
        raise ValueError('Benchmark inputs must be finite; no missing-date fill')
    exposure = stock_exposure_at_previous_open
    if not exposure.between(0, 1).all():
        raise ValueError('Exposure must be in [0, 1]')
    if (index_return <= -1).any() or (cash <= -1).any():
        raise ValueError('Return must exceed -100%')
    result = exposure * index_return + (1 - exposure) * cash
    return pd.DataFrame({'matched_return': result, 'net_value': (1 + result).cumprod(),
                         'pre_interval_stock_exposure': exposure})
