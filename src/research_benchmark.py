"""Index/cash comparison aligned to idealized close exits and open NAV marks."""
from __future__ import annotations
import numpy as np
import pandas as pd


def close_exit_matched_returns(daily:pd.DataFrame,index_open:pd.Series,index_close:pd.Series,exits:pd.DataFrame,*,initial_capital:float)->pd.Series:
    """Beginning-open stock weights, reduced at close by sold opening units.

    A day with no exit reduces exactly to lagged weight times open/open return.
    The benchmark is frictionless price-index plus zero-interest cash. The final
    isolated close-exit mark is included to match the strategy's terminal mark.
    """
    if not index_open.index.equals(daily.index) or not index_close.index.equals(daily.index):raise ValueError('Benchmark axes differ')
    if index_open.isna().any() or index_close.isna().any() or (index_open<=0).any() or (index_close<=0).any():raise ValueError('Missing benchmark prices')
    opening_value=daily.stock_value.copy()
    removed=pd.Series(0.,index=daily.index)
    if not exits.empty:
        removed=exits.groupby('date').opening_notional_sold.sum().reindex(daily.index,fill_value=0.)
        terminal=bool(exits.terminal_same_day_posting.any())
        if terminal:opening_value.iloc[-1]+=removed.iloc[-1]
    else:terminal=False
    pre_nav=daily.nav_after_open.copy()
    if 'terminal_exit_close_adjustment' in daily:
        pre_nav-=daily.terminal_exit_close_adjustment.fillna(0.)
    weight=opening_value/pre_nav
    if (weight<0).any() or (weight>1+1e-8).any():raise ValueError('Unfunded benchmark exposure')
    fraction=(removed/opening_value.where(opening_value.gt(0))).fillna(0.)
    if (fraction>1+1e-8).any():raise ValueError('Sold opening notional exceeds stock opening value')
    intraday=index_close/index_open-1
    overnight=index_open/index_close.shift(1)-1
    result=weight.shift(1)*(intraday.shift(1)+(1+intraday.shift(1))*(1-fraction.shift(1))*overnight)
    result.iloc[0]=0.
    if terminal:
        previous_nav=initial_capital if len(daily)==1 else daily.nav_after_open.iloc[-2]
        result.iloc[-1]+=removed.iloc[-1]/previous_nav*intraday.iloc[-1]
    result.name='matched_return'
    return result
