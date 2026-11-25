from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_audit import check_membership, missing_keys


def test_constant_member_count_does_not_prove_point_in_time():
    frame = pd.DataFrame({'date': pd.to_datetime(['2020-01-02']),
                          'snapshot_date':pd.to_datetime(['2019-12-31']),
                          'code':['A'], 'index_code':['000905.SH']})
    assert check_membership(frame) == ['unverified_membership_effective_and_available_dates']
    frame.snapshot_date = pd.to_datetime(['2020-01-31'])
    assert 'missing_or_future_snapshot_date' in check_membership(frame)


def test_membership_future_available_date_is_rejected():
    frame = pd.DataFrame({'date':pd.to_datetime(['2020-01-02']),
                          'snapshot_date':pd.to_datetime(['2019-12-31']),
                          'code':['A'], 'index_code':['000905.SH'],
                          'effective_date':['2020-01-02'], 'available_at':['2020-01-03'],
                          'source_event_id':['official_event']})
    assert 'future_membership_event_information' in check_membership(frame)


def test_missing_keys_retains_uncovered_difficult_samples():
    left = pd.DataFrame({'date':[1,2,2], 'code':['A','A','delisted']})
    right = pd.DataFrame({'date':[1,2], 'code':['A','A']})
    result = missing_keys(left,right,['date','code'])
    assert result.to_dict('records') == [{'date':2,'code':'delisted'}]
