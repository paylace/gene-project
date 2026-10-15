"""Verify a downloaded evidence archive and restore without overwriting changes."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    catalog = json.loads((ROOT / 'evidence/catalog.json').read_text(encoding='utf-8'))
    digest = lambda b: hashlib.sha256(b).hexdigest()
    data = args.archive.read_bytes()
    asset = next((x for x in catalog['assets'] if x['name'] == args.archive.name), None)
    if asset is None or digest(data) != asset['sha256']:
        raise ValueError('Archive not registered or SHA-256 mismatch')
    with zipfile.ZipFile(args.archive) as z:
        names = z.namelist()
        if len(set(names)) != len(names) or len(names) != asset['files']:
            raise ValueError('Invalid member count')
        for name in names:
            parts = PurePosixPath(name).parts
            target = (ROOT / name).resolve()
            if name not in catalog['files'] or '\\' in name or '..' in parts or not target.is_relative_to(ROOT):
                raise ValueError('Unsafe or unregistered path')
            b = z.read(name)
            if digest(b) != catalog['files'][name]['sha256']:
                raise ValueError('Member SHA-256 mismatch')
            if target.exists() and digest(target.read_bytes()) != digest(b):
                raise ValueError(f'Refusing to overwrite changed file: {name}')
        for name in names:
            target = ROOT / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(name))
    print(f'Verified/restored {len(names)} files: {args.archive.name}')

if __name__ == '__main__':
    main()
