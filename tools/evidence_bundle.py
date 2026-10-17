"""명시한 연구 자료만 보관하고 전체 해시 검증/없는 파일 복원을 재사용한다."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def create(archive, catalog, includes):
    archive, catalog = archive.resolve(), catalog.resolve()
    if not archive.is_relative_to(ROOT) or not catalog.is_relative_to(ROOT) or archive.exists() or catalog.exists():
        raise ValueError('보관본/목록은 작업 공간의 새 경로여야 함')
    files = set()
    for name in includes:
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.exists() or '.git' in path.parts:
            raise ValueError('허용하지 않은 포함 경로')
        for p in path.rglob('*') if path.is_dir() else (path,):
            if p.is_file() and '__pycache__' not in p.parts:
                resolved = p.resolve()
                if not resolved.is_relative_to(ROOT) or '.git' in resolved.parts:
                    raise ValueError('작업 공간 밖 파일')
                files.add(p)
    manifest = {p.relative_to(ROOT).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())}
                for p in sorted(files)}
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode('utf-8')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for name in manifest:
            bundle.write(ROOT / name, name)
        bundle.writestr('MANIFEST.json', encoded)
    record = dict(archive=archive.relative_to(ROOT).as_posix(), bytes=archive.stat().st_size,
                  sha256=sha(archive.read_bytes()), manifest_sha256=sha(encoded), files=len(manifest))
    verify(archive, record)
    catalog.parent.mkdir(parents=True, exist_ok=True)
    with catalog.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(record, ensure_ascii=False))


def verify(archive, catalog, restore=False):
    if sha(archive.read_bytes()) != catalog['sha256'] or archive.stat().st_size != catalog['bytes']:
        raise ValueError('보관본 해시/크기 불일치')
    with zipfile.ZipFile(archive) as bundle:
        encoded = bundle.read('MANIFEST.json')
        if sha(encoded) != catalog['manifest_sha256'] or bundle.testzip() is not None:
            raise ValueError('목록 해시/CRC 불일치')
        manifest = json.loads(encoded)
        if len(manifest) != catalog['files'] or len(bundle.namelist()) != len(manifest) + 1 or set(bundle.namelist()) != set(manifest) | {'MANIFEST.json'}:
            raise ValueError('파일 목록 불일치')
        for name, item in manifest.items():
            relative = PurePosixPath(name)
            path = (ROOT / name).resolve()
            if relative.is_absolute() or '..' in relative.parts or '\\' in name or not path.is_relative_to(ROOT) or '.git' in path.parts:
                raise ValueError('허용 범위 밖 복원 경로')
            data = bundle.read(name)
            if len(data) != item['bytes'] or sha(data) != item['sha256']:
                raise ValueError('파일 내용 불일치')
            if restore and path.exists() and sha(path.read_bytes()) != item['sha256']:
                raise ValueError('다른 기존 파일은 덮어쓰지 않음: ' + name)
        restored = 0
        if restore:
            for name in manifest:
                path = (ROOT / name).resolve()
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open('xb') as stream:
                        stream.write(bundle.read(name))
                    restored += 1
    return dict(passed=True, files=len(manifest), restored=restored)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('create', 'verify', 'restore-missing'))
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--include', action='append', default=[])
    args = parser.parse_args()
    if args.mode == 'create':
        create(args.archive, args.catalog, args.include)
    else:
        print(json.dumps(verify(args.archive, json.loads(args.catalog.read_bytes()), args.mode == 'restore-missing')))
