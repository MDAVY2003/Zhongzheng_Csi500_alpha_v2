from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from paired_accounting import InfeasibleSharedPath, matched_benchmark, replay_stock_only
import realistic_backtest as bt
from test_realistic_backtest_v2 import synthetic_inputs


def run_ic_inputs(inputs: dict[str, object], stop: int = 34):
    cfg = bt.BacktestConfig(strategy_mode='buffered_v2', futures_contract_rounding='round_half_up')
    basic = inputs['basic'].copy()
    basic.ts_code = basic.ts_code.str.replace('IF', 'IC')
    basic.multiplier = 200.0
    daily = inputs['futures_daily'].copy()
    daily.ts_code = daily.ts_code.str.replace('IF', 'IC')
    daily['open'] += np.tile(np.arange(len(inputs['dates'])) * 3.0, 2).reshape(2, -1).T.ravel()
    daily['settle'] = daily['open'] + 1
    settle = inputs['futures_settle'].copy()
    settle.ts_code = settle.ts_code.str.replace('IF', 'IC')
    return bt.backtest_single_factor(
        inputs['factor'], inputs['qfq_open'], inputs['raw_open'], inputs['amount'],
        inputs['market_cap'], inputs['up'], inputs['down'], inputs['suspended'],
        inputs['membership'], basic, daily, settle, factor_name='synthetic_IC',
        selected_direction=1, start_date=inputs['dates'][21], end_date=inputs['dates'][stop],
        config=cfg, qfq_close=inputs['qfq_close'])


def test_actual_engine_stock_path_cash_costs_and_future_removal():
    inputs = synthetic_inputs()
    inputs['qfq_open'].iloc[24:] *= 1.02
    inputs['qfq_close'].iloc[24:] *= 1.02
    a = run_ic_inputs(inputs)
    b = replay_stock_only(a, initial_capital=10_000_000)
    x = b['daily']
    pd.testing.assert_series_equal(x.stock_value, a['daily'].stock_value)
    for key in ['trades', 'orders']:
        pd.testing.assert_frame_equal(b[key], a[key].loc[a[key].asset_type.eq('stock')])
    pd.testing.assert_series_equal(b['holdings'].posttrade_notional, a['holdings'].posttrade_notional)
    np.testing.assert_allclose(x.nav_after_open, x.stock_value + x.cash_after_open)
    assert (x.required_margin == 0).all()
    assert (x.futures_hedge_pnl == 0).all()
    assert x.total_cost.sum() > 0
    assert a['daily'].futures_transaction_cost.sum() > 0
    assert a['rolls'].paired_roll_contracts.sum() > 0
    assert a['futures_positions'].contract.str.startswith('IC').all()
    position = a['futures_positions'].iloc[0]
    assert position.notional == pytest.approx(abs(position.contracts) * position.open_price * 200)
    assert x.iloc[0].stock_exposure < .51
    assert x.iloc[0].cash_after_open > 4_000_000
    assert x.iloc[-1].stock_market_pnl == 0


def test_future_data_cannot_change_past_trades_or_returns():
    inputs = synthetic_inputs()
    original = run_ic_inputs(inputs)
    # Change stocks, factors and prices strictly after the prefix's ending open.
    for key in ['factor','qfq_open','raw_open','qfq_close']:
        inputs[key].iloc[28:] *= 1.3
    changed = run_ic_inputs(inputs)
    pd.testing.assert_frame_equal(original['daily'].iloc[:7], changed['daily'].iloc[:7])
    truncated = run_ic_inputs(inputs, stop=27)
    pd.testing.assert_frame_equal(original['daily'].iloc[:7], truncated['daily'], check_freq=False)


def test_benchmark_matches_known_beginning_exposure_and_cash():
    dates = pd.bdate_range('2020-01-01', periods=3)
    ret = pd.Series([.1,-.1,.02], index=dates)
    exposure = pd.Series([0,.5,.6], index=dates)
    result = matched_benchmark(ret, exposure)
    np.testing.assert_allclose(result.matched_return, [0,-.05,.012])
    np.testing.assert_allclose(result.net_value, [1,.95,.95*1.012])
    with pytest.raises(ValueError):
        matched_benchmark(ret, exposure.shift(1))
    with pytest.raises(ValueError):
        matched_benchmark(ret, exposure.iloc[1:])


def test_replay_rejects_inconsistent_or_unfundable_ledger():
    a = run_ic_inputs(synthetic_inputs(), stop=24)
    a['daily'].iloc[1, a['daily'].columns.get_loc('nav_after_open')] += 1
    with pytest.raises(ValueError, match='futures leg'):
        replay_stock_only(a, initial_capital=10_000_000)
    a = run_ic_inputs(synthetic_inputs(), stop=24)
    # A's hedge gains fund stocks that B cannot fund: construct coherent ledgers.
    d = a['daily']
    d['stock_value'] = 11_000_000.
    d['stock_market_pnl'] = 0.
    d['stock_transaction_cost'] = 0.
    d['stock_slippage_cost'] = 0.
    d['stock_buy_notional'] = [11_000_000., 0., 0., 0.]
    d['stock_sell_notional'] = 0.
    d['futures_transaction_cost'] = 0.
    d['futures_submission_fee'] = 0.
    d['futures_slippage_cost'] = 0.
    d['futures_hedge_pnl'] = [2_000_000., 0., 0., 0.]
    d['nav_after_open'] = 12_000_000.
    with pytest.raises(InfeasibleSharedPath):
        replay_stock_only(a, initial_capital=10_000_000)


def test_replay_cli_writes_verifiable_same_path_outputs(tmp_path):
    import json
    import subprocess
    a = run_ic_inputs(synthetic_inputs(), stop=24)
    source = tmp_path / 'synthetic_A'
    target = tmp_path / 'synthetic_B'
    source.mkdir()
    for key in ['daily','holdings','orders','trades']:
        a[key].to_parquet(source / (key + '.parquet'))
    cli = Path(__file__).resolve().parents[1] / 'scripts/replay_stock_only.py'
    subprocess.run([sys.executable, str(cli), '--hedged-dir', str(source),
                    '--output-dir', str(target), '--initial-capital', '10000000'], check=True)
    expected = replay_stock_only(a, initial_capital=10_000_000)
    pd.testing.assert_frame_equal(pd.read_parquet(target/'daily.parquet'), expected['daily'])
    metadata = json.loads((target/'metadata.json').read_text())
    assert len(metadata['input_sha256']) == 4
    assert len(metadata['code_sha256']) == 2
    assert metadata['source_signal_and_configuration_verified'] is False
