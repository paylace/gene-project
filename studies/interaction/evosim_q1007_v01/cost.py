"""훈련 분할만으로 검증 크기 비용을 측정하고 사전 상한을 적용한다."""
import argparse
import json
import shutil
import time
from pathlib import Path
from evosim_interaction_v02.datasets import make_split
from evosim_interaction_v02.storage import read_json, write_once, file_digest
from evosim_interaction_v02.settings import digest_json
from .protocol import RESULTS, load
from .search import directory
from .evaluation import evaluate
from .audit import verify_trajectories
from .model import Treatment, NETWORK
from .network import Genome


def gate_values(search_seconds, audit_seconds, decoded_bytes, free_bytes):
    reasons = []
    if 2 * (search_seconds + audit_seconds) > 4 * 3600:
        reasons.append('실행·감사 시간의2배 추정이4시간 초과')
    if decoded_bytes > 20 * 2 ** 30:
        reasons.append('비압축 JSON 원자료 추정20GiB 초과')
    if free_bytes < 2 * decoded_bytes + 5 * 2 ** 30:
        reasons.append('디스크 여유 기준 미충족')
    return reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', type=Path, required=True)
    args = parser.parse_args()
    root = RESULTS / 'pilot'
    config = load(root)
    evidence = read_json(args.gate)
    if not evidence['passed'] or evidence['source_digest'] != config['source_digest'] or evidence['plan_digest'] != config['plan_digest']:
        raise ValueError('예비와 동일한 통과 구현 게이트 필요')
    pilot = read_json(root / 'cost_report.json')
    if not pilot['all_jobs_complete'] or pilot['known_completed_episode_calls'] != 76800:
        raise ValueError('전체 훈련 전용 예비 필요')
    destination = root / 'cost_gate.json'
    if destination.exists():
        raise FileExistsError('완료 비용 기록 보존')
    episodes = make_split().train[:252]
    lookup = {e.episode_id: e for e in episodes}
    rows = []
    for name, size in (('E20', 20), ('E40', 40)):
        cache = root / f'cost_fixture_{name}.json.gz'
        if cache.exists():
            raise FileExistsError('이전 비용 실행 보존; 재개 별도 진단 필요')
        initial = read_json(directory(root, name, 'evolution', 2901) / 'result.json')['initial_population']
        start, candidates = time.perf_counter(), []
        # 시작/완료 원장으로 비용 표본의 실제 호출과 중단 상한도 보존한다.
        work = root / f'cost_fixture_{name}_attempt.json'
        attempt = {'status': 'started', 'maximum_episode_calls': size * 756,
                   'source_digest': config['source_digest'], 'purpose': 'training-only cost fixture'}
        from evosim_interaction_v02.storage import write_json
        write_once(work, attempt)
        for slot, item in enumerate(initial):
            genome = Genome.from_dict(item, NETWORK)
            metrics, records = evaluate(genome, episodes, 2901, f'cost:q1007:v01:{name}:{slot}', Treatment('F0'), repeats=3)
            candidates.append({'genome': item, 'metrics': metrics, 'records': records})
        elapsed = time.perf_counter() - start
        decoded = len(json.dumps(candidates, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode())
        write_once(cache, candidates)
        attempt.update(status='evaluated', completed_episode_calls=size * 756,
                       single_step_actions=sum(len(r['actions']) for c in candidates for r in c['records']))
        write_json(work, attempt)
        start = time.perf_counter()
        for candidate in candidates:
            verify_trajectories(candidate['records'], lookup)
        audit_seconds = time.perf_counter() - start
        rows.append({'population': size, 'episode_calls': size * 756, 'evaluation_seconds': elapsed,
                     'audit_seconds': audit_seconds, 'decoded_json_bytes': decoded, 'stored_bytes': cache.stat().st_size})
    calls = sum(r['episode_calls'] for r in rows)
    if calls != 45360 or calls > config['plan']['cost_gate']['extra_cost_fixture_episode_slots_max']:
        raise ValueError('추가 비용 표본 예산 위반')
    training_calls, validation_calls = 38400000, 120960000
    # 전체 후보의 기록을 보존하며 gzip을 풀었을 때의 JSON 바이트로 사전 문턱을 판단한다.
    training_unit_bytes = pilot['decoded_json_bytes'] / 76800
    validation_unit_bytes = max(r['decoded_json_bytes'] / r['episode_calls'] for r in rows)
    decoded = int(training_calls * training_unit_bytes + validation_calls * validation_unit_bytes)
    evaluation_seconds = training_calls * pilot['search_seconds'] / 76800 + validation_calls * max(r['evaluation_seconds'] / r['episode_calls'] for r in rows)
    audit_seconds = training_calls * pilot['audit_seconds'] / 76800 + validation_calls * max(r['audit_seconds'] / r['episode_calls'] for r in rows)
    free = shutil.disk_usage(RESULTS).free
    reasons = gate_values(evaluation_seconds, audit_seconds, decoded, free)
    report = {'passed': not reasons, 'reasons': reasons, 'source_digest': config['source_digest'],
              'plan_digest': config['plan_digest'], 'pilot_protocol_digest': config['protocol_digest'],
              'pilot_report_sha256': file_digest(root / 'cost_report.json'), 'workers': pilot['workers'],
              'additional_training_only_episode_calls': calls, 'validation_or_test_partition_calls': 0,
              'fixtures': rows, 'estimated_search_seconds': evaluation_seconds,
              'estimated_audit_seconds': audit_seconds, 'upper_seconds_factor2': 2 * (evaluation_seconds + audit_seconds),
              'estimated_decoded_json_bytes': decoded, 'free_disk_bytes': free,
              'scope': '실측 단가의 사전 예산 외삽; 실제 본 비용 아님; 단가·시간 상한 보증 아님',
              'performance_based_selection': False, 'independent_review': False}
    write_once(destination, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if reasons:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
