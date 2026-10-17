"""고정 기술 사례의 속도·크기·동치·호출 원장을 새 폴더에 보존한다."""
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
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'studies/interaction'
sys.path.insert(0, str(STUDY))
from evosim_runtime_v01.storage import Store, TableCodec, canonical, digest
from evosim_runtime_v01.records import LineCollectRecords
from evosim_runtime_v01.accounting import COUNTS
from evosim_q1007_v02.provenance import source_manifest
from evosim_q1007_v01.resources import peak_memory_bytes

SOURCE = STUDY / 'results_q1007_v01/pilot'
BASE = STUDY / 'optimization_q1011_v01'


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical(value))


def verify_plan(out):
    plan = read(out / 'plan.json')
    if plan['sources'] != source_manifest() or plan['tool_sha256'] != file_sha(Path(__file__)):
        raise ValueError('벤치마크 고정 뒤 코드 변경')
    return plan


def evaluation_worker(out, task):
    from evosim_interaction_v02.datasets import make_split
    from evosim_interaction_v02.rollout import derive_seed
    from evosim_q1007_v01.network import initial_genome
    from evosim_q1007_v02.evaluation import evaluate_population
    case = task['case']
    seed, size = case['seed'], case['population']
    rng = random.Random(derive_seed('q1007:v01:initial', seed))
    genomes = [initial_genome(rng) for _ in range(size)]
    episodes = make_split().train[:case['episodes']]
    purposes = [(f'train:0:q1007:v01:slot{i}' if case['training'] else f'cost:q1011:slot{i}') for i in range(size)]
    start = time.perf_counter()
    results = evaluate_population(genomes, episodes, seed, purposes, repeats=case['repeats'],
                                  training=case['training'], backend=task['backend'])
    seconds = time.perf_counter() - start
    records = sum(len(r) for _, r in results)
    if records != size * case['episodes'] * case['repeats']:
        raise ValueError('평가 호출 불일치')
    result = dict(seconds=seconds, result_sha256=digest(results), policy_accounting=dict(COUNTS),
                  peak_memory_bytes=peak_memory_bytes(), episode_records=records, all_training_partition=True)
    # 전 행동·실제 지표를 보관한다. 측정 시간에는 사후 보관 비용을 넣지 않는다.
    Store(level=3).write(out / 'evaluation_records' / (task['name'] + '.json.gz'), results)
    return result


def storage_worker(out, task, plan):
    new = task['backend'] == 'tables'
    store = Store(TableCodec(LineCollectRecords()) if new else None, level=3)
    totals = dict(encode_seconds=0., restore_seconds=0., original_bytes=0, stored_bytes=0,
                  original_decoded_bytes=0, decoded_bytes=0, files=0)
    details = []
    for relative in plan['trajectory_files']:
        path = SOURCE / relative
        if file_sha(path) != plan['input_files'][relative]['sha256']:
            raise ValueError('원자료 변경')
        plain = gzip.decompress(path.read_bytes())
        value = json.loads(plain)
        target = out / 'storage_records' / task['name'] / relative
        start = time.perf_counter()
        store.write(target, value)
        encode_seconds = time.perf_counter() - start
        start = time.perf_counter()
        restored = store.read(target)
        if canonical(restored) != plain:
            raise ValueError('전체 원본 기록이 다름')
        restore_seconds = time.perf_counter() - start
        compressed = target.read_bytes()
        decoded = gzip.decompress(compressed)
        row = dict(path=relative, stored_bytes=len(compressed), original_bytes=path.stat().st_size,
                   original_decoded_bytes=len(plain), decoded_bytes=len(decoded), sha256=file_sha(target),
                   encode_seconds=encode_seconds, restore_seconds=restore_seconds)
        details.append(row)
        for key in totals:
            totals[key] += 1 if key == 'files' else row[key]
    return dict(totals=totals, files=details, all_original_bytes_equal=True,
                peak_memory_bytes=peak_memory_bytes(), policy_accounting=dict(COUNTS))


def worker(out, index):
    plan = verify_plan(out)
    task = plan['schedule'][index]
    attempt_path = out / 'attempts' / (task['name'] + '.json')
    attempt = dict(task=task, status='started', maximum_episode_calls=task.get('maximum_calls', 0))
    save(attempt_path, attempt)
    result = evaluation_worker(out, task) if task['kind'] == 'evaluation' else storage_worker(out, task, plan)
    result.update(task=task, passed=True)
    save(out / 'measurements' / (task['name'] + '.json'), result)
    # 원자 교체로 시작 기록과 완료 기록을 구분한다.
    Store().write(attempt_path, dict(attempt, status='completed', policy_accounting=result['policy_accounting']), replace=True)
    print(json.dumps({'completed': task['name'], 'seconds': result.get('seconds'),
                      'stored_bytes': result.get('totals', {}).get('stored_bytes')}), flush=True)


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    development = sorted(BASE.glob('development_*/summary.json'))
    dev = [dict(path=p.relative_to(ROOT).as_posix(), sha256=file_sha(p), **read(p)) for p in development]
    if not dev or not dev[-1]['passed']:
        raise ValueError('통과한 최신 구현 검사 필요')
    latest_sources = read(development[-1].parent / 'sources.json')
    if any(file_sha(ROOT / p) != h for p, h in latest_sources.items()):
        raise ValueError('마지막 검사 후 코드 변경')
    development_calls = sum(d['policy_accounting']['reference_episodes'] + d['policy_accounting']['optimized_episodes']
                            + d['policy_accounting']['unknown_failed_episodes_upper'] for d in dev)
    if development_calls > 100000:
        raise ValueError('개발 fixture 호출 예산 초과')
    cases = [dict(population=n, episodes=e, repeats=r, training=t, seed=3101 if n == 20 else 3102)
             for n in (20, 40) for e, r, t in ((24, 1, True), (252, 3, False))]
    schedule = []
    for repeat in range(3):
        tasks = []
        for case in cases:
            for backend in ('reference', 'fixed'):
                tasks.append(dict(kind='evaluation', case=case, backend=backend, repeat=repeat,
                    name=f"eval_p{case['population']}_e{case['episodes']}_{backend}_{repeat}",
                    maximum_calls=case['population'] * case['episodes'] * case['repeats']))
        for backend in ('json', 'tables'):
            tasks.append(dict(kind='storage', backend=backend, repeat=repeat, name=f'storage_{backend}_{repeat}'))
        random.Random(71101 + repeat).shuffle(tasks)
        schedule.extend(tasks)
    assert sum(t.get('maximum_calls', 0) for t in schedule) == 280800
    manifest = {p.relative_to(SOURCE).as_posix(): {'bytes': p.stat().st_size, 'sha256': file_sha(p)}
                for p in sorted(SOURCE.rglob('*')) if p.is_file()}
    plan = dict(experiment_id='Q1-011', created_utc=datetime.now(timezone.utc).isoformat(),
                sources=source_manifest(), tool_sha256=file_sha(Path(__file__)), development=dev,
                development_calls_upper=development_calls, schedule=schedule, input_files=manifest,
                trajectory_files=[p for p in manifest if p.endswith('.json.gz')],
                expected_benchmark_episode_calls=280800, overall_episode_calls_cap=380800,
                timing_scope='서로 다른 프로세스3회, OS 캐시 통제 안 함. 평가와 저장 별도 측정. 저장은 fsync 포함.',
                independent_review=False)
    save(out / 'plan.json', plan)
    for index in range(len(schedule)):
        subprocess.run([sys.executable, '-B', '-u', '-X', 'utf8', str(Path(__file__).resolve()),
                        '--output', str(out), '--worker', str(index)], check=True)
    reports = [read(out / 'measurements' / (t['name'] + '.json')) for t in schedule]
    evaluation, storage = {}, {}
    for case in cases:
        key = f"p{case['population']}_e{case['episodes']}"
        group = [r for r in reports if r['task']['kind'] == 'evaluation' and r['task']['case'] == case]
        if len({r['result_sha256'] for r in group}) != 1:
            raise ValueError('조건 간 실제 결과 바이트 불일치')
        row = {'case': case, 'all_results_identical': True}
        for backend in ('reference', 'fixed'):
            arm = [r for r in group if r['task']['backend'] == backend]
            row[backend] = dict(seconds_runs=[r['seconds'] for r in arm],
                               median_seconds=statistics.median(r['seconds'] for r in arm),
                               peak_memory_bytes_max=max(r['peak_memory_bytes'] for r in arm),
                               rng_objects_created=[r['policy_accounting']['rng_objects_created'] for r in arm])
        row['speedup'] = row['reference']['median_seconds'] / row['fixed']['median_seconds']
        evaluation[key] = row
    for backend in ('json', 'tables'):
        arm = [r for r in reports if r['task']['kind'] == 'storage' and r['task']['backend'] == backend]
        if not all(r['all_original_bytes_equal'] for r in arm) or len({r['totals']['stored_bytes'] for r in arm}) != 1:
            raise ValueError('저장 결과 불일치')
        storage[backend] = dict(bytes=arm[0]['totals']['stored_bytes'], decoded_bytes=arm[0]['totals']['decoded_bytes'],
            encode_seconds_runs=[r['totals']['encode_seconds'] for r in arm],
            restore_seconds_runs=[r['totals']['restore_seconds'] for r in arm],
            encode_seconds_median=statistics.median(r['totals']['encode_seconds'] for r in arm),
            restore_seconds_median=statistics.median(r['totals']['restore_seconds'] for r in arm),
            peak_memory_bytes_max=max(r['peak_memory_bytes'] for r in arm))
    accounting = {k: sum(r['policy_accounting'][k] for r in reports) for k in COUNTS}
    if accounting['reference_episodes'] + accounting['optimized_episodes'] != 280800 or accounting['unknown_failed_episodes_upper']:
        raise ValueError('실제 호출 원장이 명세와 다름')
    unchanged = all(file_sha(SOURCE / name) == meta['sha256'] for name, meta in manifest.items())
    stable = source_manifest() == plan['sources']
    if not unchanged or not stable:
        raise ValueError('실행 중 완료 소스/자료 변경')
    summary = dict(experiment_id='Q1-011', passed=True, evaluation=evaluation, storage=storage,
        benchmark_accounting=accounting, development_calls_upper=development_calls,
        all_technical_calls_upper=development_calls + 280800, sources_and_inputs_unchanged=True,
        source_digest=digest(plan['sources']), plan_sha256=file_sha(out / 'plan.json'),
        storage_reduction_fraction=1 - storage['tables']['bytes'] / storage['json']['bytes'],
        main_or_heldout_partition_calls=0, independent_review=False)
    save(out / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE / 'benchmark_01')
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError('출력은 작업 공간 안이어야 함')
    if args.worker is None:
        main(out)
    else:
        worker(out, args.worker)
