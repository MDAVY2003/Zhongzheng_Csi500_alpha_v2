"""Causal Long-priority candidate filtering; execution remains in the core engine."""
from __future__ import annotations
from dataclasses import dataclass
from math import ceil
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ShortVetoConfig:
    enabled: bool = True
    top_fraction: float = .30
    max_sell_veto_days: int = 5

    def validate(self) -> None:
        if not 0 < self.top_fraction <= 1:
            raise ValueError('Short top_fraction must be in (0,1]')
        if isinstance(self.max_sell_veto_days, bool) or not isinstance(self.max_sell_veto_days, int) or self.max_sell_veto_days < 0:
            raise ValueError('max_sell_veto_days must be a nonnegative integer')


def plan_independent_targets(*, ordered_codes: list[str], long_ranks: pd.Series,
        long_entries: list[str], normal_exits: list[str], mandatory_exits: list[str],
        pending_exits: set[str], held_codes: list[str], close_holdings: pd.Series,
        close_nav: float, execution_membership: pd.Series, short_values: pd.Series | None,
        short_membership: pd.Series, config, policy: ShortVetoConfig,
        veto_age: dict[str, int], signal_valid: bool, initial: bool,
        date: pd.Timestamp, info_date: pd.Timestamp, execute_allowed: bool,
        max_normal_replacements: int | None = None) -> dict:
    """Plan independent buy/sell sets using only pre-execution information.

    Counts concern candidate-stock-days, not hypothetical filled orders. Missing
    Short ranks veto both sides; timeout/protection may override. Age persists
    after timeout or blocked execution until Long recovery or leaving holdings.
    """
    policy.validate()
    if max_normal_replacements is not None and (
        isinstance(max_normal_replacements, bool) or not isinstance(max_normal_replacements, int)
        or max_normal_replacements < 1
    ):
        raise ValueError('max_normal_replacements must be a positive integer or None')
    held = set(held_codes)
    desired = close_holdings.copy()
    sells = {c: 'index_removal' if not execution_membership.get(c, False) else 'factor_missing_5d'
             for c in mandatory_exits}
    buys: dict[str, str] = {}
    events: list[dict] = []
    new_order: list[str] = []
    exposure = float(close_holdings.sum()/close_nav)
    protect = bool(held and (exposure < config.stock_exposure_lower or exposure > config.stock_exposure_upper))
    diagnostics = dict(v6_buy_veto_count=0, v6_sell_veto_count=0,
        v6_timeout_override_count=0, v6_exposure_override_count=0,
        v6_exposure_protection_trigger=int(protect and execute_allowed),
        v6_qualified_buy_count=0, v6_normal_sell_count=0,
        v6_normal_buy_capacity=0, v6_initial_exempt=int(initial and execute_allowed))
    if max_normal_replacements is not None:
        diagnostics.update(v6_normal_replacement_limit=max_normal_replacements,
            v6_normal_requests_before_limit=0, v6_normal_deferred_by_limit=0,
            v6_normal_limit_binding=0)
    for code in list(veto_age):
        if code not in held or code in mandatory_exits or code in pending_exits:
            veto_age.pop(code, None)
        elif signal_valid and code in long_ranks and int(long_ranks[code]) <= ceil(len(long_ranks)*config.retention_fraction):
            veto_age.pop(code, None)
    desired.loc[mandatory_exits] = 0.
    if not signal_valid or not execute_allowed:
        return dict(desired=desired,sells=sells,buys=buys,buy_order=new_order,events=events,diagnostics=diagnostics)
    base = close_nav*config.target_stock_exposure/config.target_stock_count
    if initial:
        desired[:] = 0.
        new_order = [c for c in ordered_codes if execution_membership.get(c, False)][:config.target_stock_count]
        desired.loc[new_order] = base
        buys = {c:'initial_entry' for c in new_order}
        return dict(desired=desired,sells=sells,buys=buys,buy_order=new_order,events=events,diagnostics=diagnostics)
    ranks = pd.Series(dtype='int64')
    if policy.enabled:
        finite = pd.to_numeric(short_values.where(short_membership), errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
        sorted_short = pd.DataFrame({'code':finite.index.astype(str),'score':finite.values}).sort_values(['score','code'],ascending=[False,True],kind='mergesort')
        if len(sorted_short) >= config.min_stocks:
            ranks = pd.Series(np.arange(1,len(sorted_short)+1),index=sorted_short.code,dtype='int64')
    cutoff=ceil(len(ranks)*policy.top_fraction)

    def permit(code: str, side: str) -> bool:
        if not policy.enabled:
            return True
        rank=float(ranks.get(code,np.nan))
        short_ok = np.isfinite(rank) and ((rank <= cutoff) if side=='buy' else (rank > cutoff))
        if short_ok:
            if side=='sell':veto_age.pop(code,None)
            return True
        if side=='sell':
            age=veto_age.get(code,0)+1
            veto_age[code]=age
        else:age=0
        override='exposure_protection' if protect else ('sell_timeout' if side=='sell' and age>policy.max_sell_veto_days else '')
        decision='override' if override else 'veto'
        events.append(dict(date=date,execution_info_date=info_date,code=code,side=side,
            long_rank=int(long_ranks.get(code,-1)),short_rank=rank,short_valid_count=len(ranks),
            short_cutoff=cutoff,missing_short=not np.isfinite(rank),sell_veto_age=age,
            decision=decision,override_reason=override,prior_close_stock_exposure=exposure))
        if override:
            diagnostics['v6_exposure_override_count' if protect else 'v6_timeout_override_count']+=1
        else:diagnostics[f'v6_{side}_veto_count']+=1
        return bool(override)

    # Filter in Long order. Short is never a sorting key.
    eligible_buys=[c for c in long_entries if c not in held and execution_membership.get(c,False) and permit(c,'buy')]
    eligible_sells=[c for c in normal_exits if permit(c,'sell')]
    vacancies=max(0,config.target_stock_count-len(held))
    mandatory_slots=len(mandatory_exits)
    reserve=vacancies+mandatory_slots
    capacity=max(0,len(eligible_buys)-reserve)
    requested=min(len(eligible_sells),capacity)
    allowed=requested if max_normal_replacements is None else min(requested,max_normal_replacements)
    if max_normal_replacements is not None:
        diagnostics.update(v6_normal_requests_before_limit=requested,
            v6_normal_deferred_by_limit=requested-allowed,
            v6_normal_limit_binding=int(requested>allowed))
    selected_sells=eligible_sells[:allowed]
    selected_buys=eligible_buys[:min(len(eligible_buys),reserve+len(selected_sells))]
    for code in selected_sells:
        desired[code]=0.
        sells[code]='normal_replacement'
    for i,code in enumerate(selected_buys):
        desired[code]=base
        buys[code]='vacancy_fill' if i<reserve else 'normal_replacement'
    diagnostics.update(v6_qualified_buy_count=len(eligible_buys),
        v6_normal_sell_count=len(selected_sells),v6_normal_buy_capacity=capacity)
    return dict(desired=desired,sells=sells,buys=buys,buy_order=selected_buys,events=events,diagnostics=diagnostics)
