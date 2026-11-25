"""检查 GitHub 文件白名单、公开摘要来源、链接和敏感模式；不运行研究。"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
# 仅定位疑似敏感文件和行号，不打印匹配的秘密内容。
PATTERNS={
    '本机绝对路径': re.compile(r'/(?:Users|home)/[A-Za-z0-9_.-]+/|/private/(?:var|tmp)/'),
    '私钥': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'GitHub凭据': re.compile(r'\b(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})'),
    'AWS访问键': re.compile(r'\bAKIA[A-Z0-9]{16}\b'),
    '疑似硬编码凭据': re.compile(r'(?i)(?:api_key|access_token|password|secret_key)\s*[=:]\s*["\x27][A-Za-z0-9_+/=-]{16,}["\x27]'),
}

def select_files(root: Path, config: dict) -> tuple[list[Path], list[str]]:
    files=[];errors=[]
    for name in config['root_files']:
        path=root/name
        if not path.is_file() or path.is_symlink(): errors.append(f'缺少常规根文件：{name}')
        else: files.append(path)
    for name in config['directories']:
        base=root/name
        if base.is_symlink():
            errors.append(f'公开目录不能是软链接：{name}');continue
        for path in sorted(base.rglob('*')):
            rel=path.relative_to(root)
            if any(part in config['excluded_directories'] for part in rel.parts): continue
            if path.is_symlink(): errors.append(f'公开范围发现软链接：{rel}');continue
            if path.is_file() and path.suffix in config['allowed_suffixes']:
                files.append(path)
    return sorted(set(files)),errors

def inspect(root: Path) -> tuple[list[dict],dict]:
    config=json.loads((root/'config/publication.json').read_text())
    files,errors=select_files(root,config)
    rows=[]
    relative={str(p.relative_to(root)) for p in files}
    for path in files:
        name=str(path.relative_to(root));data=path.read_bytes()
        if len(data)>config['max_file_bytes']:errors.append(f'超出公开体积上限：{name}')
        rows.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        try:text=data.decode('utf-8-sig')
        except UnicodeDecodeError:
            errors.append(f'非UTF-8文本：{name}');continue
        for kind,pattern in PATTERNS.items():
            for match in pattern.finditer(text):errors.append(f'{kind}：{name}:{text.count(chr(10),0,match.start())+1}')
        if path.suffix=='.md':
            # 只检查真实 Markdown 链接；代码块中的本地运行路径不当作公开链接。
            text=re.sub(r'```.*?```','',text,flags=re.S)
            for target in re.findall(r'\]\(([^\s)]+)\)',text):
                if target.startswith(('https:','http:','mailto:','#')):continue
                target=unquote(target.split('#')[0])
                dest=(path.parent/target).resolve()
                try:rel=str(dest.relative_to(root.resolve()))
                except ValueError:
                    errors.append(f'链接指向仓库外：{name}');continue
                if dest.is_file() and rel not in relative:errors.append(f'链接目标未公开：{name} -> {rel}')
                elif not dest.exists():errors.append(f'失效链接：{name} -> {rel}')
    # 公开摘要的内容哈希在任何无数据副本上都可以核对。
    provenance=json.loads((root/'reports/source_manifest.json').read_text())
    for record in provenance['files']:
        p=root/'reports'/record['public_path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=record['public_sha256']:
            errors.append(f'公开结果与来源清单不匹配：{record["public_path"]}')
    git_checked=False
    if (root/'.git').exists():
        # 只读 Git；不修改用户索引。
        output=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root)
        git_files={s for s in output.decode().split('\0') if s}
        extras=sorted(git_files-relative)
        missing=sorted(relative-git_files)
        errors += [f'Git候选超出白名单：{p}' for p in extras]
        errors += [f'白名单文件被Git忽略：{p}' for p in missing]
        git_checked=True
    result={'files':len(rows),'total_bytes':sum(r['bytes'] for r in rows),'max_file_bytes':max((r['bytes'] for r in rows),default=0),
        'git_candidates_checked':git_checked,'public_source_records':len(provenance['files']),'errors':errors,
        'scope':'白名单、Git候选、大小、常见敏感模式、Markdown本地链接及公开摘要哈希；不代表穷尽所有秘密类型或数据授权审核。'}
    return rows,result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='未存在的本地检查目录')
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    rows,result=inspect(ROOT)
    with (out/'github_files.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['errors']:raise SystemExit(1)
if __name__=='__main__':main()
