"""예비·본·조건부 확인의 작업 실행과 감사."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
import gzip
from evosim_interaction_v02.storage import read_json, write_json, write_once
from .protocol import prepare, RESULTS, load
from .search import run_search, directory, attempt_accounting
from .audit import audit_job
from .resources import peak_memory_bytes


def measured_search(root, job, pause_after):
    result = run_search(root, **job, pause_after=pause_after)
    write_json(directory(root, **job) / f'resources_{pause_after or "complete"}.json',
               {'peak_worker_memory_bytes': peak_memory_bytes(), 'scope': '프로세스 수명 전체의 최대 작업 집합; 동시 전체 최대값 아님'})
    return result


def execute(root, config, workers, pause_after=None):
    errors = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(measured_search, root, job, pause_after): job for job in config['jobs']}
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
                print(f'DONE {job}', flush=True)
            except Exception as exc:
                errors.append({'job': job, 'error': repr(exc)})
                write_json(Path(root) / 'failures.json', errors)
    if errors:
        raise RuntimeError('기술 실패 원본 보존; 검사 후 별도 재개 필요')


def audit_all(root, config, workers):
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(audit_job, root, **job): job for job in config['jobs']}
        for future in as_completed(futures):
            results.append(future.result())
            print(f'AUDIT {len(results)}/{len(futures)}', flush=True)
    return sorted(results, key=lambda r: (r['condition'], r['method'], r['seed']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('pilot', 'main', 'confirmation'))
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--cost', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        raise ValueError('비용 계획은 최대 작업자2개')
    root = RESULTS / args.phase
    if (root / 'failures.json').exists():
        raise ValueError('기존 기술 실패를 먼저 확인할 것')
    config = prepare(root, args.phase, args.gate, cost=args.cost)
    start = time.perf_counter()
    if args.phase == 'pilot':
        execute(root, config, args.workers, pause_after=7)
    execute(root, config, args.workers)
    search_seconds = time.perf_counter() - start
    start = time.perf_counter()
    audits = audit_all(root, config, args.workers)
    audit_seconds = time.perf_counter() - start
    accounting = [attempt_accounting(directory(root, **job)) for job in config['jobs']]
    calls = sum(r['known_completed_episode_calls'] for r in accounting)
    unknown = sum(r['interrupted_attempts'] for r in accounting)
    if unknown:
        raise ValueError('미확정 원장; 판정 중단')
    write_once(root / 'search_audit.json', {'passed': True, 'jobs': audits, 'calls': calls,
                                           'search_seconds': search_seconds, 'audit_seconds': audit_seconds})
    if args.phase == 'pilot':
        complete = len(audits) == 6 and calls == 76800
        files = [p for p in root.rglob('*') if p.is_file()]
        report = {'all_jobs_complete': complete, 'source_digest': config['source_digest'],
                  'plan_digest': config['plan_digest'], 'protocol_digest': config['protocol_digest'],
                  'search_seconds': search_seconds, 'audit_seconds': audit_seconds,
                  'known_completed_episode_calls': calls, 'unknown_attempts': unknown,
                  'decoded_json_bytes': sum(len(gzip.decompress(p.read_bytes())) if p.suffix == '.gz' else p.stat().st_size for p in files),
                  'stored_bytes': sum(p.stat().st_size for p in files), 'validation_or_test_calls': 0,
                  'audits': audits, 'workers': args.workers, 'performance_based_selection': False}
        write_once(root / 'cost_report.json', report)
        if not complete:
            raise ValueError('예비 구성/호출 수 불일치')
    else:
        from .finalize import lock_all, evaluate_locked
        lock_all(root)
        result = evaluate_locked(root)
        print(result['decision'], flush=True)


if __name__ == '__main__':
    main()
