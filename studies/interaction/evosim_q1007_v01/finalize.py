"""모든 예산 후보 잠금 뒤에만 테스트하며 실제 호출 원장을 보존한다."""
from pathlib import Path
from datetime import datetime, timezone
import uuid
from evosim_interaction_v02.storage import read_json, write_json, write_once, file_digest
from evosim_interaction_v02.settings import digest_json
from evosim_interaction_v02.datasets import make_split
from evosim_interaction_v02.audit import check_metrics
from .protocol import load
from .network import Genome, NETWORK
from .model import Treatment
from .search import directory, validate_checkpoint, attempt_accounting
from .evaluation import evaluate
from .audit import verify_trajectories
from .analysis import decision


def candidate_manifest(root):
    config = load(root)
    if config['phase'] not in ('main', 'confirmation') or config['fixture']:
        raise ValueError('예비/합성 실행의 테스트 잠금 금지')
    jobs = {}
    for job in config['jobs']:
        shape = config['job_shapes'][job['condition']]
        path = directory(root, **job)
        state, result = read_json(path / 'checkpoint.json'), read_json(path / 'result.json')
        validate_checkpoint(state)
        if state['next_round'] != shape['rounds'] or state['result_digest'] != digest_json(result) or state['identity'] != result['identity'] or result['identity']['protocol_digest'] != config['protocol_digest']:
            raise ValueError('미완료 또는 변경된 실행')
        if set(state['record_hashes']) != {f'round_{g:03d}.json.gz' for g in range(shape['rounds'])}:
            raise ValueError('회차 목록 불완전')
        if any(file_digest(path / name) != value for name, value in state['record_hashes'].items()):
            raise ValueError('회차 원자료 변경')
        if set(result['endpoints']) != set(shape['endpoint_rounds'].values()):
            raise ValueError('끝점 불완전')
        for end, snapshot in result['endpoints'].items():
            candidate = snapshot['candidate']
            if candidate['selection_source'] != 'validation' or candidate['round'] >= snapshot['through_round'] or candidate['digest'] != Genome.from_dict(candidate['genome'], NETWORK).digest:
                raise ValueError('끝점 밖 후보 또는 검증 미선발')
        jobs[f"{job['condition']}/{job['method']}_{job['seed']}"] = {'result_sha256': file_digest(path / 'result.json'), 'endpoints': result['endpoints']}
    return {'protocol_digest': config['protocol_digest'], 'jobs': jobs}


def lock_all(root):
    manifest = candidate_manifest(root)
    manifest['lock_digest'] = digest_json(manifest)
    write_once(Path(root) / 'candidate_lock.json', manifest)
    return manifest


def verify_lock(root):
    locked = read_json(Path(root) / 'candidate_lock.json')
    if digest_json({k: v for k, v in locked.items() if k != 'lock_digest'}) != locked['lock_digest']:
        raise ValueError('후보 잠금 체크섬 오류')
    if candidate_manifest(root) != {k: v for k, v in locked.items() if k != 'lock_digest'}:
        raise ValueError('잠금 뒤 후보 변경')
    return locked


def test_candidate(root, config, locked, candidate, seed):
    identity = {'protocol_digest': config['protocol_digest'], 'lock_digest': locked['lock_digest'],
                'seed': seed, 'genome_digest': candidate['digest'], 'purpose': 'test:q1007:v01', 'repeats': 3}
    path = Path(root) / 'test_cache' / f'{seed}_{candidate["digest"]}.json.gz'
    if not path.exists():
        attempt = Path(root) / 'test_work/attempts' / f'{uuid.uuid4().hex}.json'
        record = {'status': 'started', 'identity': identity, 'maximum_episode_calls': 1512,
                  'maximum_single_step_actions': 1512 * 24, 'started_utc': datetime.now(timezone.utc).isoformat()}
        write_json(attempt, record)
        metrics, records = evaluate(Genome.from_dict(candidate['genome'], NETWORK), make_split().test,
                                    seed, identity['purpose'], Treatment('F0'), repeats=3)
        payload = {'metrics': metrics, 'records': records}
        write_once(path, {'identity': identity, 'payload': payload, 'payload_digest': digest_json(payload)})
        record.update(status='evaluated', completed_episode_calls=len(records), completed_single_step_actions=sum(len(r['actions']) for r in records))
        write_json(attempt, record)
    value = read_json(path)
    if value['identity'] != identity or value['payload_digest'] != digest_json(value['payload']):
        raise ValueError('테스트 캐시 식별자/체크섬 오류')
    return value['payload'], path


def evaluate_locked(root):
    config = load(root)
    locked = verify_lock(root)
    rates, details, seen = {}, [], set()
    lookup = {ep.episode_id: ep for ep in make_split().test}
    expected = [(ep.episode_id, rep) for rep in range(3) for ep in make_split().test]
    for seed in config['seeds']:
        for job in config['jobs']:
            if job['seed'] != seed:
                continue
            key = f"{job['condition']}/{job['method']}_{seed}"
            for label, snapshot in locked['jobs'][key]['endpoints'].items():
                candidate = snapshot['candidate']
                payload, cache = test_candidate(root, config, locked, candidate, seed)
                if [(r['episode_id'], r['repeat']) for r in payload['records']] != expected:
                    raise ValueError('테스트 분할/반복/순서 불일치')
                check_metrics(payload['metrics'], payload['records'])
                verify_trajectories(payload['records'], lookup)
                rates.setdefault(label, []).append(payload['metrics']['success_rate'])
                reused = (seed, candidate['digest']) in seen
                seen.add((seed, candidate['digest']))
                details.append({'seed': seed, 'endpoint': label, 'candidate_digest': candidate['digest'],
                                'metrics': payload['metrics'], 'cache_reused': reused,
                                'cache': cache.relative_to(root).as_posix(), 'cache_sha256': file_digest(cache)})
    if any(len(v) != 80 for v in rates.values()):
        raise ValueError('끝점의 대응 시드 불완전')
    active = None
    if config['phase'] == 'confirmation':
        active = read_json(Path(root).parent / 'main/summary.json')['decision']['eligible']
    result = {'rates': rates, 'details': details, 'decision': decision(rates, active=active),
              'actual_test_accounting': attempt_accounting(Path(root) / 'test_work'),
              'logical_test_slots': len(details) * 1512, 'unique_test_genomes': len(seen),
              'lock_digest': locked['lock_digest'], 'protocol_digest': config['protocol_digest'],
              'source_digest': config['source_digest']}
    if result['actual_test_accounting']['interrupted_attempts']:
        raise ValueError('미확정 중단 호출 보존; 판정 중단')
    if result['actual_test_accounting']['known_completed_episode_calls'] != len(seen) * 1512:
        raise ValueError('실제 테스트 호출 원장 불일치')
    write_once(Path(root) / 'summary.json', result)
    return result
