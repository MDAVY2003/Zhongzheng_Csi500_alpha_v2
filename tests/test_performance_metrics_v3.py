import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from performance_metrics import max_drawdown


def test_max_drawdown_includes_initial_nav():
    series = pd.Series([0.9, 0.9])
    mdd, _ = max_drawdown(series)
    assert abs(mdd + 0.10) < 1e-12


def test_explicit_new_runs_root_is_used(tmp_path, monkeypatch):
    import json
    import sys
    import performance_metrics as pm
    new = tmp_path / 'corrected'
    run = new / 'v3_bucket4_nd3'
    run.mkdir(parents=True)
    pd.DataFrame([{'variant':'v3_bucket4_nd3'}]).to_csv(new / 'comparison.csv', index=False)
    (run / 'daily.parquet').touch()
    (run / 'benchmarks.parquet').touch()
    seen=[]
    def fake_metrics(path):
        seen.append(path)
        row={key:0. for key in ['cagr','annualized_vol','sharpe','sortino','calmar','max_drawdown',
            'information_ratio','active_tstat','gross_sharpe','cost_drag_cagr','annualized_turnover','average_stock_exposure']}
        row.update(start='2018-01-02',end='2022-12-30')
        return row,pd.Series([0.],index=pd.to_datetime(['2018-01-31']))
    monkeypatch.setattr(pm,'metrics_for',fake_metrics)
    out=tmp_path/'metrics'
    monkeypatch.setattr(sys,'argv',[pm.__file__,'--runs-root',str(new),'--output',str(out)])
    pm.main()
    assert seen==[run]
    assert len(json.loads((out/'performance_summary.json').read_text()))==1
