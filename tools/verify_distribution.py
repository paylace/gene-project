"""Restore an isolated distribution and verify every registered evidence byte."""
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'dist/github'
TARGET = OUT / 'restore-check'

def main():
    if TARGET.exists():
        raise SystemExit('Use a fresh restore-check directory; existing verification is preserved.')
    TARGET.mkdir()
    with zipfile.ZipFile(OUT / 'gene-project-source.zip') as z:
        z.extractall(TARGET)
    catalog = json.loads((TARGET / 'evidence/catalog.json').read_text(encoding='utf-8'))
    for asset in catalog['assets']:
        subprocess.run([sys.executable, '-B', '-X', 'utf8', 'tools/restore_evidence.py', str(OUT / asset['name'])], cwd=TARGET, check=True)
    for name, item in catalog['files'].items():
        assert hashlib.sha256((TARGET / name).read_bytes()).hexdigest() == item['sha256'], name
    subprocess.run([sys.executable, '-B', '-X', 'utf8', 'tools/check_research_layout.py', '--require-data'], cwd=TARGET, check=True)
    result = {'passed': True, 'restored_files': len(catalog['files']), 'purpose': 'distribution round trip; no experiments rerun'}
    (ROOT / 'docs/management/distribution_check.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
