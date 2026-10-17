"""물리 실행별 설정·단계별 접근·동결 프로토콜."""
from pathlib import Path
import platform
import sys
import numpy as np
from evosim_interaction_v02.datasets import make_split
from evosim_interaction_v02.settings import digest_json
from evosim_interaction_v02.storage import read_json, write_once, manifest_digest, file_digest
from .model import Treatment
from .source import ROOT, source_manifest, plan

RESULTS = ROOT / 'studies/interaction/results_q1007_v01'


def shapes(phase):
    rounds = 20 if phase == 'pilot' else 200
    return {'E20': {'population': 20, 'rounds': rounds, 'method': 'evolution',
                    'endpoint_rounds': {} if phase == 'pilot' else {'100': 'B', '200': 'T'}},
            'E40': {'population': 40, 'rounds': rounds, 'method': 'evolution',
                    'endpoint_rounds': {} if phase == 'pilot' else {'100': 'P', '200': 'PT'}},
            'R20': {'population': 20, 'rounds': 20 if phase == 'pilot' else 400, 'method': 'random',
                    'endpoint_rounds': {} if phase == 'pilot' else {'100': 'R2000', '200': 'R4000', '400': 'R8000'}}}


def runtime():
    return {'python': platform.python_version(), 'numpy': np.__version__, 'platform': sys.platform}


def save_config(root, phase, seeds, job_shapes, *, fixture=False):
    value = {'experiment_id': 'Q1-007', 'phase': phase, 'fixture': fixture,
             'source_digest': manifest_digest(source_manifest()), 'sources': source_manifest(),
             'plan_digest': digest_json(plan()), 'plan': plan(), 'datasets_digest': make_split().digest,
             'runtime': runtime(), 'training_batch': 24, 'repeats': 3,
             'job_shapes': job_shapes, 'seeds': seeds,
             'jobs': [{'condition': name, 'method': s['method'], 'seed': seed}
                      for name, s in job_shapes.items() for seed in seeds],
             'settings': {name: Treatment('F0').to_dict() for name in job_shapes}}
    value['protocol_digest'] = digest_json(value)
    write_once(Path(root) / 'protocol.json', value)
    return value


def fixture_config(root, *, size=2, rounds=20, endpoints=None, method='evolution', validation=True):
    if not 2 <= size <= 40 or not 1 <= rounds <= 100:
        raise ValueError('합성 검증 크기 초과')
    return save_config(root, 'verification', [41], {'F': {'population': size, 'rounds': rounds,
        'method': method, 'validation_rounds': list(range(4, rounds, 5)) if validation else [],
        'endpoint_rounds': endpoints or {str(rounds): 'fixture'}}}, fixture=True)


def prepare(root, phase, gate, *, cost=None):
    root = Path(root).resolve()
    if phase not in ('pilot', 'main', 'confirmation') or root != (RESULTS / phase).resolve():
        raise ValueError('새 공식 출력 경로만 허용')
    evidence = read_json(gate)
    if not evidence.get('passed') or not evidence.get('full_gate') or evidence['source_digest'] != manifest_digest(source_manifest()) or evidence['plan_digest'] != digest_json(plan()) or evidence['runtime'] != runtime():
        raise ValueError('현재 소스·계획·환경의 통과 게이트 필요')
    job_shapes = shapes(phase)
    if phase != 'pilot':
        cost_path = Path(cost) if cost else RESULTS / 'pilot/cost_gate.json'
        report = read_json(cost_path)
        pilot = load(RESULTS / 'pilot')
        if not report.get('passed') or report['source_digest'] != evidence['source_digest'] or report['plan_digest'] != evidence['plan_digest'] or report['pilot_protocol_digest'] != pilot['protocol_digest'] or report['pilot_report_sha256'] != file_digest(RESULTS / 'pilot/cost_report.json'):
            raise ValueError('일치하는 훈련 전용 예비·비용 통과 필요')
        if not read_json(RESULTS / 'pilot/cost_report.json')['all_jobs_complete']:
            raise ValueError('예비 미완료')
    if phase == 'confirmation':
        from .analysis import decision
        summary = read_json(RESULTS / 'main/summary.json')
        expected = decision(summary['rates'])
        if expected != summary['decision'] or not expected['eligible']:
            raise ValueError('본 판정이 허용한 확인 없음')
        active = set(expected['eligible'])
        job_shapes.pop('R20')
        if active == {'PT-B'}:
            job_shapes['E20'].update(rounds=100, endpoint_rounds={'100': 'B'})
            job_shapes['E40']['endpoint_rounds'] = {'200': 'PT'}
        elif active == {'P-T'}:
            job_shapes['E20']['endpoint_rounds'] = {'200': 'T'}
            job_shapes['E40'].update(rounds=100, endpoint_rounds={'100': 'P'})
    return save_config(root, phase, plan()[phase + '_seeds'], job_shapes)


def load(root, condition=None):
    config = read_json(Path(root) / 'protocol.json')
    if config['protocol_digest'] != digest_json({k: v for k, v in config.items() if k != 'protocol_digest'}):
        raise ValueError('프로토콜 체크섬 불일치')
    if config['source_digest'] != manifest_digest(source_manifest()) or config['plan_digest'] != digest_json(plan()) or config['runtime'] != runtime() or config['datasets_digest'] != make_split().digest:
        raise ValueError('실행 소스·계획·환경·분할 불일치')
    if condition is not None:
        config = dict(config, **config['job_shapes'][condition])
        config['validation_rounds'] = config.get('validation_rounds', [] if config['phase'] == 'pilot' else list(range(9, config['rounds'], 10)))
    return config


def validation_episodes(config):
    split = make_split()
    return split.train[:2] if config['fixture'] else split.validation


def check_access(config, purpose, episodes):
    split = make_split()
    allowed = split.train if purpose.startswith('train:') else validation_episodes(config) if purpose == 'validation' else ()
    if config['phase'] == 'pilot' and purpose == 'validation':
        allowed = ()
    if not episodes or not {ep.episode_id for ep in episodes} <= {ep.episode_id for ep in allowed}:
        raise ValueError('단계별 자료 접근 위반')
