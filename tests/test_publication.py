"""公开导出不修改源文件，也不能带入交易明细或秘密。"""
import hashlib
import json
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_public_release import export,sanitize
from check_publication import PATTERNS,select_files


def test_export_preserves_only_final_results_and_excludes_history(tmp_path):
    from prepare_public_release import FINAL_EXPERIMENTS
    originals={}
    for label,(relative,variants) in FINAL_EXPERIMENTS.items():
        stage=tmp_path/relative;stage.mkdir(parents=True)
        original='variant,net_cumulative\n'+'\n'.join(f'{v},-0.01' for v in variants)+'\n'
        (stage/'comparison.csv').write_text(original)
        originals[label]=(stage,original)
        (stage/'daily.parquet').write_bytes(b'private')
        (stage/'report.md').write_text('不导出额外报告')
    legacy=tmp_path/'results/v3_old';legacy.mkdir()
    (legacy/'comparison.csv').write_text('variant,net_cumulative\nold,0.9\n')
    out=tmp_path/'public'
    export(tmp_path,out)
    for label,(stage,original) in originals.items():
        assert (stage/'comparison.csv').read_text()==original
        assert (out/label/'comparison.csv').read_text()==original
    assert not list(out.rglob('*.parquet')) and not list(out.rglob('*.md'))
    assert not (out/'v3_old').exists()
    records=json.loads((out/'source_manifest.json').read_text())['files']
    assert len(records)==2
    for record in records:
        assert record['source_sha256']==hashlib.sha256((tmp_path/record['source']).read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):export(tmp_path,out)


def test_final_export_rejects_missing_variant(tmp_path):
    from prepare_public_release import FINAL_EXPERIMENTS
    relative,variants=FINAL_EXPERIMENTS['v3_is']
    stage=tmp_path/relative;stage.mkdir(parents=True)
    (stage/'comparison.csv').write_text('variant\n'+variants[0]+'\n')
    with pytest.raises(ValueError,match='四组'):export(tmp_path,tmp_path/'public')


def test_paths_are_redacted_without_changing_relative_data(tmp_path):
    text=sanitize('/'+'Users/example/project/file '+str(tmp_path)+'/data/raw.csv',tmp_path)
    assert '/Users/' not in text and 'data/raw.csv' in text


def test_sensitive_patterns_detect_tokens_without_network():
    value='ghp_'+'a'*36
    assert PATTERNS['GitHub凭据'].search(value)
    assert PATTERNS['私钥'].search('-----BEGIN '+'PRIVATE KEY-----')


def test_public_whitelist_rejects_symlink(tmp_path):
    (tmp_path/'src').mkdir(); (tmp_path/'data').mkdir()
    (tmp_path/'data/private.py').write_text('x=1')
    (tmp_path/'src/link.py').symlink_to(tmp_path/'data/private.py')
    files,errors=select_files(tmp_path,{'root_files':[],'directories':['src'],'excluded_directories':['data'],'allowed_suffixes':['.py']})
    assert not files and errors
