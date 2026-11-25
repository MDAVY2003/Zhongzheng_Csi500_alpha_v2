from pathlib import Path
import sys
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from research_benchmark import close_exit_matched_returns


def test_close_sale_excludes_next_overnight_and_no_sale_matches_open_return():
    dates=pd.bdate_range('2024-01-01',periods=3)
    o=pd.Series([100.,120.,132.],index=dates);c=pd.Series([110.,126.,132.],index=dates)
    d=pd.DataFrame({'stock_value':[50.,50.,50.],'nav_after_open':[100.,100.,100.]},index=dates)
    no=close_exit_matched_returns(d,o,c,pd.DataFrame(),initial_capital=100.)
    np.testing.assert_allclose(no,[0,.1,.05])
    exits=pd.DataFrame({'date':[dates[0]],'opening_notional_sold':[50.],'terminal_same_day_posting':[False]})
    out=close_exit_matched_returns(d,o,c,exits,initial_capital=100.)
    np.testing.assert_allclose(out,[0,.05,.05])


def test_final_day_exit_close_mark_is_included():
    dates=pd.bdate_range('2024-01-01',periods=2)
    o=pd.Series([100.,100.],index=dates);c=pd.Series([100.,110.],index=dates)
    d=pd.DataFrame({'stock_value':[50.,0.],'nav_after_open':[100.,105.],'terminal_exit_close_adjustment':[np.nan,5.]},index=dates)
    exits=pd.DataFrame({'date':[dates[-1]],'opening_notional_sold':[50.],'terminal_same_day_posting':[True]})
    result=close_exit_matched_returns(d,o,c,exits,initial_capital=100.)
    np.testing.assert_allclose(result,[0,.05])
