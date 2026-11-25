"""将最终组合的面板、方向和窗口绑定到同一次因子审计。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioFactorSource:
    factor: str
    path: Path
    audit_path: Path
    provenance: str
    direction: float
    bucket: str
    window: int


def _flag(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if str(value).strip().lower() in {'true', 'false'}:
        return str(value).strip().lower() == 'true'
    raise ValueError(f'无法解析因子来源标记：{value!r}')


def resolve_portfolio_sources(
    selected: pd.DataFrame, *, base_audit: Path, strict_audit: Path,
    base_factor_dir: Path, strict_factor_dir: Path,
) -> dict[str, PortfolioFactorSource]:
    """仅消费已入选结果；严格复核来源不完整时失败，不退回基础面板。"""
    if selected.empty or selected['factor'].duplicated().any():
        raise ValueError('入选因子为空或重复')
    base = pd.read_csv(base_audit).set_index('factor')
    strict = pd.read_csv(strict_audit).set_index('factor') if strict_audit.exists() else pd.DataFrame()
    if base.index.has_duplicates or strict.index.has_duplicates:
        raise ValueError('因子审计记录重复')
    sources = {}
    for rec in selected.to_dict('records'):
        name = rec['factor']
        if not _flag(rec.get('final_eligible', False)):
            raise ValueError(f'{name}未通过最终资格审核')
        base_exact = name in base.index and _flag(base.loc[name].get('is_signal_anchor_exact', False))
        needs_strict = _flag(rec.get('strict_available', False)) or not base_exact
        if needs_strict:
            if name not in strict.index:
                raise ValueError(f'{name}缺少 strict 审计，禁止退回 approximate')
            audit = strict.loc[name]
            audit_path = strict_audit
            path = strict_factor_dir / f'{name}.parquet'
            provenance = 'strict'
        else:
            audit = base.loc[name]
            audit_path = base_audit
            path = base_factor_dir / f'{name}.parquet'
            provenance = 'base_exact'
        if not path.is_file():
            raise FileNotFoundError(f'{name}缺少 {provenance} 因子面板：{path}；禁止近似回退')
        if audit.get('status') != 'success' or not _flag(audit.get('is_signal_anchor_exact', False)):
            raise ValueError(f'{name}实际来源不是已成功计算的 exact/strict 审计')
        direction = float(audit['direction'])
        bucket = str(audit['bucket'])
        if direction not in {-1., 1.} or direction != float(rec['direction']) or bucket != rec['bucket']:
            raise ValueError(f'{name}入选方向/分组与{provenance}审计不一致')
        value = 1 if bucket == 'short' else audit['lifetime' if bucket in {'middle', 'long'} else 'lifetime_width']
        if bucket not in {'short', 'middle', 'long', 'delayed'} or not np.isfinite(value) or value < 1 or value != int(value):
            raise ValueError(f'{name}审计窗口无效：{value}')
        sources[name] = PortfolioFactorSource(name, path.resolve(), audit_path.resolve(), provenance, direction, bucket, int(value))
    return sources
