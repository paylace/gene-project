"""관측 열 codec의 추가 비교. 기존 결과/코드 보존, 새 정책 호출0."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'studies/interaction'
sys.path.insert(0, str(STUDY))
from evosim_runtime_v02.storage import Store, TableCodec, canonical, digest
from evosim_q1007_v02.provenance import source_manifest
from evosim_q1007_v01.resources import peak_memory_bytes

SOURCE = STUDY / 'results_q1007_v01/pilot'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    return {**source_manifest(), **{p.relative_to(ROOT).as_posix(): sha(p)
        for p in (STUDY / 'evosim_runtime_v02').rglob('*.py')}, 'tool': sha(Path(__file__))}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical(value))


def worker(out, index):
    plan = json.loads((out / 'plan.json').read_bytes())
    if sources() != plan['sources']:
        raise ValueError('측정 코드 변경')
    task = plan['schedule'][index]
    store = Store(TableCodec() if task['codec'] else None, level=task['level'])
    details = []
    for name in plan['trajectory_files']:
        source = SOURCE / name
        if sha(source) != plan['input_files'][name]['sha256']:
            raise ValueError('완료 원본 변경')
        plain = gzip.decompress(source.read_bytes())
        original = json.loads(plain)
        path = out / 'records' / task['name'] / name
        start = time.perf_counter()
        store.write(path, original)
        encode_seconds = time.perf_counter() - start
        start = time.perf_counter()
        restored = store.read(path)
        if canonical(restored) != plain:
            raise ValueError('원본 전체 바이트 불일치')
        restore_seconds = time.perf_counter() - start
        raw = path.read_bytes()
        details.append(dict(path=name, bytes=len(raw), decoded_bytes=len(gzip.decompress(raw)),
                            sha256=sha(path), encode_seconds=encode_seconds, restore_seconds=restore_seconds))
    result = dict(task=task, files=details, totals={k: sum(r[k] for r in details)
        for k in ('bytes', 'decoded_bytes', 'encode_seconds', 'restore_seconds')},
        all_original_bytes_equal=True, policy_episode_calls=0, peak_memory_bytes=peak_memory_bytes())
    save(out / 'measurements' / (task['name'] + '.json'), result)
    print(json.dumps(dict(completed=task['name'], totals=result['totals'])), flush=True)


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'tests.log').open('w', encoding='utf-8') as stream:
        suite = unittest.defaultTestLoader.discover(str(STUDY / 'evosim_runtime_v02/tests'))
        tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not tests.wasSuccessful():
        print((out / 'tests.log').read_text(encoding='utf-8'))
        raise RuntimeError('기술 검사 미통과; 출력 보존')
    schedule = []
    for repeat in range(3):
        tasks = [dict(name=f'{kind}_{repeat}', codec=kind != 'json3', level=level, repeat=repeat, kind=kind)
                 for kind, level in (('json3', 3), ('columns3', 3), ('columns9', 9))]
        random.Random(71102 + repeat).shuffle(tasks)
        schedule.extend(tasks)
    manifest = {p.relative_to(SOURCE).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                for p in sorted(SOURCE.rglob('*')) if p.is_file()}
    plan = dict(experiment_id='Q1-011', version='storage-v0.2', sources=sources(), tests=tests.testsRun,
        input_files=manifest, trajectory_files=[n for n in manifest if n.endswith('.json.gz')],
        schedule=schedule, policy_episode_calls=0, independent_review=False,
        prior_benchmark_sha256=sha(STUDY / 'optimization_q1011_v01/benchmark_01/summary.json'))
    save(out / 'plan.json', plan)
    for index in range(len(schedule)):
        subprocess.run([sys.executable, '-B', '-u', '-X', 'utf8', str(Path(__file__).resolve()),
                        '--output', str(out), '--worker', str(index)], check=True)
    measurements = [json.loads(p.read_bytes()) for p in sorted((out / 'measurements').glob('*.json'))]
    groups = {}
    for kind in ('json3', 'columns3', 'columns9'):
        group = [m for m in measurements if m['task']['kind'] == kind]
        assert len(group) == 3 and all(m['all_original_bytes_equal'] for m in group)
        assert len({m['totals']['bytes'] for m in group}) == 1
        groups[kind] = dict(bytes=group[0]['totals']['bytes'], decoded_bytes=group[0]['totals']['decoded_bytes'],
            encode_seconds_median=statistics.median(m['totals']['encode_seconds'] for m in group),
            restore_seconds_median=statistics.median(m['totals']['restore_seconds'] for m in group),
            encode_seconds_runs=[m['totals']['encode_seconds'] for m in group],
            restore_seconds_runs=[m['totals']['restore_seconds'] for m in group],
            peak_memory_bytes_max=max(m['peak_memory_bytes'] for m in group))
    unchanged = all(sha(SOURCE / n) == v['sha256'] for n, v in manifest.items())
    assert unchanged and sources() == plan['sources']
    for group in groups.values():
        group['physical_reduction_fraction'] = 1 - group['bytes'] / groups['json3']['bytes']
    summary = dict(passed=True, experiment_id='Q1-011', tests=tests.testsRun, groups=groups,
                   source_and_inputs_unchanged=unchanged, policy_episode_calls=0,
                   source_digest=digest(plan['sources']), plan_sha256=sha(out / 'plan.json'), independent_review=False)
    save(out / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=STUDY / 'optimization_q1011_v02')
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError('작업 공간 출력만 허용')
    if args.worker is None:
        main(out)
    else:
        worker(out, args.worker)
