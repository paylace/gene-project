"""Q1-007 소스와 두 선행 동결 의존성을 함께 잠근다."""
import hashlib
from pathlib import Path
from evosim_q1006_v02.source import source_manifest as inherited_sources
from evosim_interaction_v02.storage import manifest_digest, read_json

ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = ROOT / 'docs/protocols/q1_007_v01_draft.json'
INHERITED_DIGEST = 'd8d2d533a4f8ad4d1eec1c8a394b5150d29e64e1cc58dc1260533c6b40511320'


def source_manifest():
    dependency = inherited_sources()
    if manifest_digest(dependency) != INHERITED_DIGEST:
        raise ValueError('완료한 Q1-006 의존 소스가 바뀜')
    here = Path(__file__).parent
    own = {f'q1007/{p.relative_to(here).as_posix()}': hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(here.rglob('*.py'))}
    return {**{f'inherited/{k}': v for k, v in dependency.items()}, **own}


def plan():
    value = read_json(PLAN_PATH)
    exact = {'experiment_id': 'Q1-007', 'main_seeds': list(range(2001, 2081)),
             'confirmation_seeds': list(range(2101, 2181)), 'pilot_seeds': [2901, 2902],
             'training_batch': 24, 'validation_every_rounds': 10, 'validation_at_round0': False,
             'evaluation_repeats': 3, 'discovery_test_success_rate': .8,
             'evolution_endpoints': {'B': [20, 100], 'P': [40, 100], 'T': [20, 200], 'PT': [40, 200]}}
    if any(value.get(k) != v for k, v in exact.items()):
        raise ValueError('사전 설계 조건 불일치')
    return value
