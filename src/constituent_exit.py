"""User-authorized idealized constituent exit. No future price selection."""
from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
import numpy as np

@dataclass(frozen=True)
class ExitValuation:
    proceeds_before_cost: float
    market_pnl: float
    reference_price: float
    price_source: str
    stale_close: bool
    impact_bp: float
    missing_adv_fixed_slippage_only: bool


def value_exit(*,holding_notional:float,mark_basis:float,current_close:float,
               latest_close:float,raw_close:float,prior_adv:float,
               impact_coefficient:float) -> ExitValuation:
    def valid(x):return np.isfinite(x) and x>0
    if not valid(holding_notional) or not valid(mark_basis):
        raise ValueError('Held position requires positive notional and carried mark')
    if valid(current_close):price=current_close;source='current_close'
    elif valid(latest_close):price=latest_close;source='latest_valid_close_at_or_before_execution'
    else:price=mark_basis;source='carried_position_mark_no_historical_close'
    proceeds=holding_notional*price/mark_basis
    adv_ok=valid(prior_adv)
    impact=impact_coefficient*sqrt(proceeds/prior_adv) if adv_ok else 0.
    return ExitValuation(proceeds,proceeds-holding_notional,raw_close if valid(raw_close) else price,
                         source,source!='current_close',impact,not adv_ok)
