"""Q1-007 보관본의 전체 해시를 검사하고 요청 시 없는 파일만 복원한다."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--restore-missing', action='store_true')
    args = parser.parse_args()
    catalog = json.loads((ROOT / 'evidence/q1_007_v01.json').read_text(encoding='utf-8'))
    if hashlib.sha256(args.archive.read_bytes()).hexdigest() != catalog['sha256']:
        raise ValueError('보관본 SHA-256 불일치')
    restored = 0
    with zipfile.ZipFile(args.archive) as archive:
        encoded = archive.read('MANIFEST.json')
        if hashlib.sha256(encoded).hexdigest() != catalog['manifest_sha256'] or archive.testzip() is not None:
            raise ValueError('목록 해시/ZIP CRC 오류')
        manifest = json.loads(encoded)
        if len(manifest) != catalog['files'] or set(archive.namelist()) != set(manifest) | {'MANIFEST.json'}:
            raise ValueError('파일 목록 불일치')
        # 전부 검사한 뒤에만 쓰며 기존에 다른 파일은 덮어쓰지 않는다.
        for name, item in manifest.items():
            relative = PurePosixPath(name)
            path = (ROOT / name).resolve()
            if relative.is_absolute() or '..' in relative.parts or '\\' in name or not path.is_relative_to(ROOT):
                raise ValueError('허용 범위 밖 경로')
            data = archive.read(name)
            if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('내부 파일 오류')
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError(f'기존 파일 내용 다름: {name}')
        if args.restore_missing:
            for name in manifest:
                path = (ROOT / name).resolve()
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(archive.read(name))
                    restored += 1
    print(json.dumps({'passed': True, 'files': len(manifest), 'restored': restored}, ensure_ascii=False))


if __name__ == '__main__':
    main()
