"""组合入口必须绑定严格面板及其审计参数，不能按文件是否存在降级。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import run_v3_portfolio_is as portfolio
import run_v3_variants_is as variants


class StopBeforeBacktest(Exception):
    pass


def fixture_inputs(tmp_path, monkeypatch, module, *, name='alpha058', strict=True, missing=False):
    dates = pd.bdate_range('2017-12-01', periods=30)
    codes = [f'{n:06}.SZ' for n in range(20)]
    values = pd.DataFrame(np.arange(600).reshape(30, 20), index=dates, columns=codes, dtype=float)
    mem = values.notna()
    selected = pd.DataFrame([dict(factor=name, direction=1., bucket='middle', d10_h=.1,
        final_eligible=True, is_signal_anchor_exact=True, strict_available=strict)])
    base = pd.DataFrame([dict(factor=name, direction=1., bucket='middle', lifetime=5,
        lifetime_width=5, status='success', is_signal_anchor_exact=not strict)])
    strict_life = base.copy() if strict else base.iloc[:0].copy()
    if strict:
        strict_life['lifetime'] = 4
        strict_life['lifetime_width'] = 4
        strict_life['is_signal_anchor_exact'] = True
    for rel, df in [('results/v3_screen_strict/selected_factors.csv', selected),
                    ('results/v3_factor_audit/d10_lifetime.csv', base),
                    ('results/v3_strict_audit/d10_lifetime.csv', strict_life)]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    base_dir = tmp_path / 'data/processed/alphas_point_in_time'
    strict_dir = tmp_path / 'data/processed/alphas_point_in_time_strict_v3'
    base_dir.mkdir(parents=True)
    strict_dir.mkdir(parents=True)
    values.to_parquet(base_dir / f'{name}.parquet')
    if strict and not missing:
        (values + 1).to_parquet(strict_dir / f'{name}.parquet')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'load_panels', lambda: dict(membership=mem, qfq_open=values,
        amount=values+1000, circulating_market_cap=values+10000))
    monkeypatch.setattr(sys, 'argv', [module.__file__, '--output', str(tmp_path / 'output'),
        '--screen', str(tmp_path / 'results/v3_screen_strict/selected_factors.csv')])
    windows, reads = [], []
    original_read = pd.read_parquet
    def track_read(path, *args, **kwargs):
        reads.append(Path(path))
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(pd, 'read_parquet', track_read)
    def track_window(pct, window, members):
        windows.append(window)
        return pct
    monkeypatch.setattr(module, 'trailing_rank', track_window)
    def stop(*args, **kwargs):
        raise StopBeforeBacktest
    monkeypatch.setattr(module, 'run_one', stop)
    return windows, reads


@pytest.mark.parametrize('module', [portfolio, variants])
def test_strict_missing_must_stop_before_reading_base(tmp_path, monkeypatch, module):
    _, reads = fixture_inputs(tmp_path, monkeypatch, module, missing=True)
    with pytest.raises(FileNotFoundError, match='strict|严格'):
        module.main()
    assert not any('alphas_point_in_time/' in str(p) for p in reads)


@pytest.mark.parametrize('module', [portfolio, variants])
def test_strict_lifetime_and_panel_have_same_source(tmp_path, monkeypatch, module):
    windows, reads = fixture_inputs(tmp_path, monkeypatch, module)
    with pytest.raises(StopBeforeBacktest):
        module.main()
    assert windows == [4]
    assert any('alphas_point_in_time_strict_v3' in str(p) for p in reads)


@pytest.mark.parametrize('module', [portfolio, variants])
def test_exact_base_remains_usable_without_strict_panel(tmp_path, monkeypatch, module):
    windows, reads = fixture_inputs(tmp_path, monkeypatch, module, name='alpha040', strict=False)
    with pytest.raises(StopBeforeBacktest):
        module.main()
    assert windows == [5]
    assert any('alphas_point_in_time/' in str(p) for p in reads)

@pytest.mark.parametrize('module', [portfolio, variants])
def test_selected_strict_direction_mismatch_stops(tmp_path, monkeypatch, module):
    _, reads = fixture_inputs(tmp_path, monkeypatch, module)
    path = tmp_path / 'results/v3_strict_audit/d10_lifetime.csv'
    df = pd.read_csv(path); df['direction'] = -1.; df.to_csv(path, index=False)
    with pytest.raises(ValueError, match='方向|分组'):
        module.main()
    assert not reads

@pytest.mark.parametrize('module', [portfolio, variants])
def test_missing_strict_audit_does_not_grant_base_eligibility(tmp_path, monkeypatch, module):
    _, reads = fixture_inputs(tmp_path, monkeypatch, module)
    path = tmp_path / 'results/v3_strict_audit/d10_lifetime.csv'
    df = pd.read_csv(path); df.iloc[:0].to_csv(path, index=False)
    with pytest.raises(ValueError, match='strict'):
        module.main()
    assert not reads
