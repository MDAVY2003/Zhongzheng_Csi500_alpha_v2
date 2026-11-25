"""Frozen temporal boundaries for CSI500 research (no OOS labels in IS)."""
from __future__ import annotations
import pandas as pd
from factor_analysis import calculate_signal_aligned_open_returns

def in_sample_open_returns(open_price: pd.DataFrame,horizon:int=1) -> pd.DataFrame:
    prices=open_price.loc['2018-01-01':'2022-12-31']
    return calculate_signal_aligned_open_returns(prices,horizon=horizon)
