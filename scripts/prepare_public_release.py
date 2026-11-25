"""只导出最终 V3 IS 与探索 OOS 摘要，不导出历史实验或生成额外报告。"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL_EXPERIMENTS = {
    'v3_is': ('results/v3_strict_window_fix_20261122_01/portfolio_is',
              ('v3_equal_nd3', 'v3_bucket4_nd3', 'v3_equal_nd5', 'v3_bucket4_nd5')),
    'v3_oos': ('results/v3_variants_oos',
               ('top15_short_middle_equal_nd3', 'top15_short_middle_equal_nd5',
                'new_bucket_75_15_5_5_nd3', 'new_bucket_75_15_5_5_nd5')),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sanitize(text: str, root: Path) -> str:
    prefixes = [(str(root/'data'), 'data'), (str((root/'data').resolve()), 'data'), (str(root), '.')]
    for origin, target in sorted(prefixes, key=lambda item: -len(item[0])):
        text = text.replace(origin, target)
    return re.sub(r'(?:/Users/|/home/|/private/)[^\s"\x27`<>]*', '[本机路径已移除]', text)


def export(root: Path, output: Path) -> dict:
    # 先核对指定的完整四组集合，不能因收益方向或文件缺失静默少报。
    sources = []
    for label, (relative, expected) in FINAL_EXPERIMENTS.items():
        source = root/relative
        with (source/'comparison.csv').open(encoding='utf-8-sig') as f:
            variants = [row['variant'] for row in csv.DictReader(f)]
        if len(variants) != len(expected) or set(variants) != set(expected):
            raise ValueError(f'{label}必须恰好包含指定的四组结果')
        sources.append((source/'comparison.csv', Path(label)/'comparison.csv'))
        for variant in expected:
            for name in ['performance.csv', 'yearly.csv', 'yearly_matched.csv', 'factor_weights.csv']:
                path=source/variant/name
                if path.is_file():
                    sources.append((path, Path(label)/variant/name))
    output.mkdir(parents=True, exist_ok=False)
    manifest=[]
    for source, rel in sources:
        target=output/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        original=source.read_text(encoding='utf-8-sig')
        public=sanitize(original,root)
        target.write_text(public,encoding='utf-8')
        manifest.append({'source':str(source.relative_to(root)), 'source_sha256':sha(source),
                         'public_path':str(rel), 'public_sha256':sha(target), 'transformed':public!=original})
    (output/'source_manifest.json').write_text(json.dumps({'description':'仅最终IS四组与探索OOS四组，来源文件未改写；本机路径脱敏。','files':manifest},ensure_ascii=False,indent=2)+'\n')
    return {'experiments':2,'variants':8,'exported_files':len(manifest),'output':str(output)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(export(ROOT,args.output.resolve()),ensure_ascii=False))

if __name__=='__main__':
    main()
