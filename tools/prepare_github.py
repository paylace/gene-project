"""Build a small checkout and immutable, hashed evidence archives. Originals stay put."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'dist' / 'github'
EXPORT = OUT / 'gene-project-ready'
SKIP = {'.git', '__pycache__', '.verification_tmp', 'dist', '.venv', 'node_modules'}

def digest(data):
    return hashlib.sha256(data).hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def tracked(p):
    rel = p.relative_to(ROOT)
    if any(part in SKIP for part in rel.parts):
        return False
    if rel.parts[0] == 'archive':
        return rel.as_posix() == 'archive/README.md'
    if p.suffix in {'.py', '.md', '.yml', '.yaml', '.csv', '.svg'}:
        return True
    if rel.parts[0] in {'docs', 'evidence', '.github'}:
        return p.suffix != '.log' and p.suffix != '.png'
    if len(rel.parts) == 1:
        return p.name in {'.gitignore', '.gitattributes', 'requirements.txt'}
    # Keep small definitions and aggregate summaries, not run/checkpoint records.
    return p.suffix == '.json' and p.name in {
        'protocol.json', 'summary.json', 'datasets.json', 'candidate_lock.json',
        'audit.json', 'diagnostics.json', 'behavior_summary.json', 'baseline_results.json'
    } and p.stat().st_size < 1_000_000

def main():
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and not any(x in SKIP for x in p.relative_to(ROOT).parts))
    raw = [p for p in files if p.relative_to(ROOT).parts[0] in {'studies', 'archive'} and not tracked(p)
           and '__pycache__' not in p.parts and 'generated_cache' not in p.parts
           and 'reference_materials' not in p.parts and p.suffix != '.pyc']
    catalog = {'schema': 1, 'release_url': 'https://github.com/paylace/gene-project/releases/tag/evidence-2026-10-15',
               'files': {p.relative_to(ROOT).as_posix(): {'size': p.stat().st_size, 'sha256': digest(p.read_bytes())} for p in raw}, 'assets': []}
    OUT.mkdir(parents=True, exist_ok=True)
    for group in ('working_memory', 'interaction', 'archive'):
        members = [p for p in raw if (p.relative_to(ROOT).parts[0] == 'archive' if group == 'archive' else p.relative_to(ROOT).parts[:2] == ('studies', group))]
        asset = OUT / f'{group}-evidence-2026-10-15.zip'
        with zipfile.ZipFile(asset, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in members:
                z.write(p, p.relative_to(ROOT).as_posix())
        with zipfile.ZipFile(asset) as z:
            assert z.testzip() is None
            for name in z.namelist():
                assert digest(z.read(name)) == catalog['files'][name]['sha256']
        catalog['assets'].append({'name': asset.name, 'size': asset.stat().st_size, 'sha256': digest(asset.read_bytes()), 'files': len(members)})
    save(ROOT / 'evidence/catalog.json', catalog)
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and tracked(p))
    # Exact ignore list avoids accidentally excluding new source or new protocols.
    ignored = sorted('/' + p.relative_to(ROOT).as_posix() for p in raw)
    (ROOT / '.gitignore').write_text('__pycache__/\n*.py[cod]\n.venv/\ndist/\n.verification_tmp/\n*.log\n/archive/\n!/archive/README.md\n' + '\n'.join(ignored) + '\n', encoding='utf-8')
    files = sorted(set(files + [ROOT / '.gitignore']))
    desired = {p.relative_to(ROOT).as_posix() for p in files}
    if EXPORT.exists():
        extra = [p.relative_to(EXPORT).as_posix() for p in EXPORT.rglob('*') if p.is_file()
                 and p.relative_to(EXPORT).as_posix() not in desired
                 and '.verification_tmp' not in p.parts and '__pycache__' not in p.parts
                 and p.suffix != '.log']
        if extra:
            raise RuntimeError(f'Export contains unexpected files; preserve and choose a fresh output: {extra[:5]}')
    for p in files:
        dest = EXPORT / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    exported = {p.relative_to(ROOT).as_posix(): {'sha256': digest(p.read_bytes()), 'size': p.stat().st_size} for p in files}
    save(OUT / 'export_manifest.json', exported)
    with zipfile.ZipFile(OUT / 'gene-project-source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
    print(json.dumps({'source_files': len(files), 'source_bytes': sum(p.stat().st_size for p in files), 'evidence_files': len(raw), 'assets': catalog['assets']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
