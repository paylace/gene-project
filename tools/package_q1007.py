"""Q1-007 v0.1의 비용 미통과 예비·소스를 손실 없이 보관한다."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'studies/interaction'))


def main():
    from evosim_q1007_v01.source import source_manifest
    from evosim_interaction_v02.storage import manifest_digest
    gate = json.loads((ROOT / 'docs/management/q1_007_execution_gate.json').read_text(encoding='utf-8'))
    sources = source_manifest()
    if manifest_digest(sources) != gate['source_digest']:
        raise ValueError('실행 소스 변경')
    files = set((ROOT / 'studies/interaction/results_q1007_v01/pilot').rglob('*'))
    files = {p for p in files if p.is_file()}
    for name, expected in sources.items():
        if name.startswith('inherited/dependency/'):
            path = ROOT / 'studies/interaction/evosim_interaction_v02' / name.removeprefix('inherited/dependency/')
        elif name.startswith('inherited/q1006/'):
            path = ROOT / 'studies/interaction/evosim_q1006_v02' / name.removeprefix('inherited/q1006/')
        else:
            path = ROOT / 'studies/interaction/evosim_q1007_v01' / name.removeprefix('q1007/')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('소스 경로/해시 불일치')
        files.add(path)
    files.update(ROOT / name for name in ('docs/protocols/q1_007_v01_draft.json',
        'docs/protocols/Q1-007_탐색예산_실험계획.md', 'docs/management/q1_007_execution_gate.json',
        'docs/management/q1_007_execution_gate.log', 'docs/management/q1_007_preparation_record.json'))
    manifest = {p.relative_to(ROOT).as_posix(): {'bytes': p.stat().st_size,
                 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)}
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
    destination = ROOT / 'dist/q1007_v01_pilot_2026-10-17.zip'
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        raise FileExistsError('기존 보관본을 덮어쓰지 않음')
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in manifest:
            archive.write(ROOT / name, name)
        archive.writestr('MANIFEST.json', encoded)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC 오류')
        for name, value in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != value['sha256']:
                raise ValueError('보관본 파일 해시 불일치')
    catalog = {'experiment_id': 'Q1-007', 'version': 'q1007-v0.1', 'status': '예비·비용 완료; 본 미실행',
               'archive': destination.relative_to(ROOT).as_posix(), 'bytes': destination.stat().st_size,
               'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(), 'files': len(manifest),
               'manifest_sha256': hashlib.sha256(encoded).hexdigest(), 'source_digest': gate['source_digest'],
               'zip_crc_and_member_hashes_verified': True, 'release_url': None}
    path = ROOT / 'evidence/q1_007_v01.json'
    if path.exists():
        raise FileExistsError('기존 보관 목록 보존')
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(catalog, ensure_ascii=False))


if __name__ == '__main__':
    main()
