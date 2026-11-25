"""Replay a completed A ledger; never creates an unrun IC experiment."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from paired_accounting import replay_stock_only
from data_audit import sha256
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hedged-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--initial-capital', type=float, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Output directory must be empty')
    source = {k: pd.read_parquet(args.hedged_dir / (k + '.parquet'))
              for k in ['daily', 'holdings', 'orders', 'trades']}
    output = replay_stock_only(source, initial_capital=args.initial_capital)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in output.items():
        frame.to_parquet(args.output_dir / (name + '.parquet'))
    (args.output_dir / 'metadata.json').write_text(json.dumps({
        'status': 'stock_path_replay', 'source': str(args.hedged_dir.resolve()),
        'initial_capital': args.initial_capital, 'cash_interest': 0,
        'weight_policy': 'same_stock_notionals_percentages_use_own_NAV',
        'command': sys.argv,
        'dates': [str(output['daily'].index.min().date()), str(output['daily'].index.max().date())],
        'input_sha256': {k: sha256(args.hedged_dir / (k + '.parquet')) for k in source},
        'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [ROOT/'src/paired_accounting.py', ROOT/'scripts/replay_stock_only.py']},
        'source_signal_and_configuration_verified': False,
    }, indent=2) + '\n')

if __name__ == '__main__':
    main()
