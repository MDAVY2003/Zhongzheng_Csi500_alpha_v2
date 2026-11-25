from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from constituent_exit import value_exit
from paired_accounting import replay_stock_only
import realistic_backtest as bt
from test_realistic_backtest_v2 import synthetic_inputs


def run(inputs,stop=28,interval=1):
    return bt.backtest_single_factor(inputs['factor'],inputs['qfq_open'],inputs['raw_open'],inputs['amount'],inputs['market_cap'],inputs['up'],inputs['down'],inputs['suspended'],inputs['membership'],inputs['basic'],inputs['futures_daily'],inputs['futures_settle'],factor_name='exit_test',selected_direction=1,start_date=inputs['dates'][21],end_date=inputs['dates'][stop],config=bt.BacktestConfig(strategy_mode='buffered_v2',futures_contract_rounding='round_half_up',idealized_constituent_exit=True,rebalance_interval=interval),qfq_close=inputs['qfq_close'],raw_close=inputs['qfq_close'])


def test_exit_at_scheduled_close_ignores_all_constraints_and_buffer():
    x=synthetic_inputs();dates=x['dates'];code='S099'
    x['membership'].loc[dates[22]:,code]=False
    x['suspended'].loc[dates[22]:,code]=True
    x['qfq_close'].at[dates[23],code]=9.
    x['down'].loc[dates[22]:,code]=9.
    a=run(x,interval=2)
    ex=a['constituent_exits']
    assert len(ex)==1
    assert ex.iloc[0].date==dates[23]
    assert ex.iloc[0].reference_close==9.
    assert ex.iloc[0].fill_ratio==1
    assert ex.iloc[0].transaction_cost>0
    assert ex.iloc[0].posting_date==dates[24]
    assert not ((a['trades'].code==code)&(a['trades'].side=='sell')&(a['trades'].date<dates[23])).any()
    # Entry still at t0 open, close sale not usable at that same morning.
    assert a['daily'].at[dates[23],'stock_value']>a['daily'].at[dates[23],'stock_value_after_exit_close']
    b=replay_stock_only(a,initial_capital=10_000_000)
    np.testing.assert_allclose(b['daily'].stock_value,a['daily'].stock_value)


def test_missing_close_uses_last_past_close_never_future_and_terminal_posted():
    x=synthetic_inputs();dates=x['dates'];code='S099'
    x['membership'].loc[dates[25]:,code]=False
    x['qfq_close'].loc[dates[25]:,code]=np.nan
    x['qfq_open'].loc[dates[25]:,code]=np.nan
    x['raw_open'].loc[dates[25]:,code]=np.nan
    a=run(x,stop=25)
    r=a['constituent_exits'].iloc[0]
    assert r.price_source=='latest_valid_close_at_or_before_execution'
    assert r.reference_close==10.
    assert r.terminal_same_day_posting
    assert a['daily'].iloc[-1].terminal_exit_close_adjustment<0
    replay_stock_only(a,initial_capital=10_000_000)
    x['qfq_close'].at[dates[26],code]=1000.
    altered=run(x,stop=25)
    pd.testing.assert_frame_equal(a['daily'],altered['daily'])


def test_no_historical_close_falls_back_to_carried_mark_with_no_adv():
    r=value_exit(holding_notional=100.,mark_basis=10.,current_close=np.nan,latest_close=np.nan,raw_close=np.nan,prior_adv=np.nan,impact_coefficient=20.)
    assert r.proceeds_before_cost==100
    assert r.market_pnl==0
    assert r.missing_adv_fixed_slippage_only
    assert r.price_source=='carried_position_mark_no_historical_close'


def test_idealized_exit_costs_reconcile_on_actual_posting_dates():
    x=synthetic_inputs();dates=x['dates']
    x['membership'].loc[dates[24]:,'S099']=False
    x['qfq_close'].at[dates[24],'S099']=12.
    a=run(x,stop=28)
    trades=a['trades'][a['trades'].asset_type.eq('stock')].copy()
    posting=trades.get('posting_date').fillna(trades.date)
    costs=trades.groupby(posting).transaction_cost.sum().reindex(a['daily'].index,fill_value=0)
    np.testing.assert_allclose(costs,a['daily'].stock_transaction_cost,atol=1e-6)
    b=replay_stock_only(a,initial_capital=10_000_000)
    assert a['constituent_exits'].iloc[0].market_pnl>0
    assert b['daily'].at[dates[25],'stock_market_pnl']>0


def test_normal_suspended_sell_remains_blocked():
    x=synthetic_inputs();dates=x['dates']
    x['factor'].loc[dates[22]:,:]=-np.arange(100.)
    x['suspended'].loc[dates[23]:,'S099']=True
    a=run(x,stop=25)
    orders=a['orders']
    blocked=orders[(orders.code=='S099')&(orders.side=='sell')]
    assert len(blocked)>0
    assert blocked.filled_notional.eq(0).all()
    assert a['constituent_exits'].empty
