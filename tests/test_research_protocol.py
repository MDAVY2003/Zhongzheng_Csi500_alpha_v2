from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from research_protocol import in_sample_open_returns

def test_is_labels_do_not_touch_oos_even_for_decay():
    dates=pd.bdate_range('2022-12-01','2023-01-31')
    p=pd.DataFrame({'A':range(100,100+len(dates))},index=dates,dtype=float)
    for h in [1,5,10]:
        a=in_sample_open_returns(p,h)
        p.loc['2023-01-01':,'A']=1e9
        b=in_sample_open_returns(p,h)
        pd.testing.assert_frame_equal(a,b)
        assert a.tail(h+1).isna().all().all()
