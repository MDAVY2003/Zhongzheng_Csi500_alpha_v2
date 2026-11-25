"""Offline data validation. Missing provenance never silently becomes PIT proof."""
from __future__ import annotations

import hashlib
from pathlib import Path
import pandas as pd
import numpy as np


DATASETS = {
    'membership': 'processed/universe/csi500_daily_membership.parquet',
    'snapshots': 'raw/index_membership/csi500/index_weight_history.parquet',
    'calendar': 'raw/index_membership/csi500/trade_calendar.parquet',
    'bars': 'processed/daily_bars.parquet',
    'cap': 'raw/factor_dependencies/daily_market_cap.parquet',
    'industry': 'raw/factor_dependencies/sw_industry_membership.parquet',
    'limits': 'raw/trading_constraints/daily_stock_limits.parquet',
    'suspensions': 'raw/trading_constraints/daily_suspensions.parquet',
    'futures_basic': 'raw/futures/ic/fut_basic.parquet',
    'futures_daily': 'raw/futures/ic/fut_daily.parquet',
    'futures_settle': 'raw/futures/ic/fut_settle.parquet',
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def missing_keys(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Return every missing key, retaining difficult samples for reporting."""
    merged = left[keys].drop_duplicates().merge(right[keys].drop_duplicates(), on=keys, how='left', indicator=True)
    return merged.loc[merged._merge.eq('left_only'), keys]


def check_membership(frame: pd.DataFrame) -> list[str]:
    issues = []
    if frame.duplicated(['date','code']).any():
        issues.append('duplicate_membership_keys')
    if 'snapshot_date' not in frame or (frame.snapshot_date > frame.date).any():
        issues.append('missing_or_future_snapshot_date')
    if 'index_code' not in frame or not frame.index_code.eq('000905.SH').all():
        issues.append('wrong_or_missing_index_code')
    # A snapshot date alone is not publication or event-effective-date evidence.
    if not {'effective_date','available_at','source_event_id'}.issubset(frame.columns):
        issues.append('unverified_membership_effective_and_available_dates')
    else:
        if frame[['effective_date','available_at','source_event_id']].isna().any().any():
            issues.append('missing_membership_event_provenance')
        if (pd.to_datetime(frame.effective_date) > frame.date).any() or (pd.to_datetime(frame.available_at) > frame.date).any():
            issues.append('future_membership_event_information')
    return issues


def audit_data(root: Path, *, approved_policy: dict | None = None) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    report: dict[str, object] = {'status': 'BLOCKED', 'datasets': {}, 'blockers': [], 'limitations': []}
    tables = {}
    for name, relative in DATASETS.items():
        path = root / relative
        if not path.is_file():
            report['blockers'].append('missing_dataset:' + name)
            continue
        frame = pd.read_parquet(path)
        for key in ['date','trade_date','snapshot_date','in_date','out_date','list_date','delist_date']:
            if key in frame:
                frame[key] = pd.to_datetime(frame[key], errors='raise')
        tables[name] = frame
        date_key = next((x for x in ['date','trade_date','snapshot_date'] if x in frame), None)
        keys = (['date','code','suspend_type','suspend_timing'] if name == 'suspensions'
                else ['trade_date','ts_code'] if 'trade_date' in frame else ['date','code'])
        report['datasets'][name] = {
            'path': str(Path('data') / relative), 'rows': len(frame), 'bytes': path.stat().st_size,
            'sha256': sha256(path), 'columns': list(frame.columns),
            'date_min': str(frame[date_key].min().date()) if date_key else None,
            'date_max': str(frame[date_key].max().date()) if date_key else None,
            'unique_codes': int(frame.code.nunique()) if 'code' in frame else None,
            'duplicate_keys': int(frame.duplicated(keys).sum()) if set(keys).issubset(frame.columns) else None,
        }
        if report['datasets'][name]['duplicate_keys']:
            report['blockers'].append('duplicate_keys:' + name)
    details = {}
    if 'membership' in tables:
        members = tables['membership'].loc[tables['membership'].is_member].copy()
        report['blockers'].extend(check_membership(members))
        counts = members.groupby('date').code.nunique()
        report['membership_counts'] = {'min': int(counts.min()),'max': int(counts.max()),'trading_days': len(counts)}
        for name in ['bars','cap','limits']:
            if name not in tables:
                continue
            valid = tables[name]
            cols = {'bars':['open','high','low','close','volume','amount'],
                    'cap':['cap','circulating_market_cap'], 'limits':['up_limit','down_limit']}[name]
            values = valid[cols].apply(pd.to_numeric, errors='coerce')
            good = np.isfinite(values).all(axis=1)
            good &= values.gt(0).all(axis=1)
            missing = missing_keys(members, valid.loc[good], ['date','code'])
            details['membership_missing_' + name] = missing
            by_code = missing.groupby('code').size().sort_values(ascending=False)
            report['membership_missing_' + name] = {
                'stock_days': len(missing),'stock_count': len(by_code),
                'by_code': {str(k): int(v) for k,v in by_code.items()},
                'interpretation': 'missing/invalid market rows; suspension or source gaps require explicit handling',
            }
            if len(missing):
                report['blockers'].append('unresolved_member_day_coverage:' + name)
    if 'futures_basic' in tables:
        basic = tables['futures_basic']
        report['futures_spec'] = {'products': sorted(basic.fut_code.dropna().unique().tolist()),
                                  'multipliers': sorted(basic.multiplier.dropna().unique().tolist())}
        if not basic.fut_code.eq('IC').all() or not basic.multiplier.gt(0).all():
            report['blockers'].append('invalid_IC_contract_spec')
    if {'futures_settle','futures_daily'}.issubset(tables):
        settle, daily = tables['futures_settle'], tables['futures_daily']
        report['futures_parameter_values'] = {col: sorted(pd.to_numeric(settle[col], errors='coerce').dropna().unique().tolist()) for col in ['trading_fee_rate','short_margin_rate','offset_today_fee']}
        missing = missing_keys(daily,settle,['trade_date','ts_code'])
        report['futures_missing_settle_rows'] = len(missing)
        if len(missing):
            report['blockers'].append('missing_futures_settlement_parameters')
        report['blockers'].append('unverified_futures_parameter_units_and_preopen_availability')
    # No guessed prices, official limit dates, industry publication dates or delisting proceeds.
    report['blockers'].extend(['missing_verified_spot_index_price_and_total_return_basis',
                              'missing_verified_IC_historical_limits_or_documented_validated_rule',
                              'unverified_industry_publication_timing',
                              'unverified_corporate_actions_and_delisting_disposition'])
    report['limitations'].extend(['stock_st_not_available_and_source_strategy_has_no_ST_filter',
                                  'source_engine_uses_total_return_notional_stock_account',
                                  'source_engine_future_settlement_parameters_include_same_day_data',
                                  'historical_metadata_may_contain_source_project_absolute_paths'])
    if approved_policy and approved_policy.get('execution_authorized'):
        accepted_prefixes = ('unresolved_member_day_coverage:', 'unverified_corporate_actions_and_delisting_disposition',
            'unverified_futures_parameter_units_and_preopen_availability', 'unverified_industry_publication_timing',
            'unverified_membership_effective_and_available_dates', 'missing_verified_IC_historical_limits_or_documented_validated_rule')
        accepted = [x for x in report['blockers'] if x.startswith(accepted_prefixes)]
        report['blockers'] = [x for x in report['blockers'] if x not in accepted]
        report['accepted_research_approximations'] = accepted
        report['assumption_version'] = approved_policy.get('assumption_version')
        if (root/'raw/benchmark/csi500_sina_price.parquet').exists():
            report['blockers'] = [x for x in report['blockers'] if x != 'missing_verified_spot_index_price_and_total_return_basis']
            report['limitations'].append('benchmark_is_price_index_dividend_basis_differs_from_stock_account')
    report['blockers'] = sorted(set(report['blockers']))
    report['status'] = 'BLOCKED' if report['blockers'] else 'READY_FOR_RESEARCH'
    return report, details
