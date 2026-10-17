"""동결 과학적 정의를 재사용하고 공통 평가 기반을 연결한다."""
from evosim_q1007_v01.model import NETWORK, Treatment
from evosim_q1007_v01.evaluation import evaluate as original, training_metrics
from evosim_interaction_v02.evaluation import summarize
from evosim_runtime_v01.linecollect import evaluate_population as engine
from evosim_runtime_v01.accounting import completed, COUNTS
from evosim_interaction_v02.datasets import make_split


def reference(genome, episodes, seed, purpose, *, repeats=1, training=False):
    maximum = len(episodes) * repeats
    try:
        result = original(genome, episodes, seed, purpose, Treatment('F0'), repeats=repeats, training=training)
        completed('reference', [result])
        return result
    except Exception:
        COUNTS['unknown_failed_episodes_upper'] += maximum
        raise


def metrics(records, training):
    return training_metrics(records, Treatment('F0')) if training else summarize(records)


def evaluate_population(genomes, episodes, seed, purposes, *, repeats=1, training=False, backend='auto'):
    episodes, purposes = tuple(episodes), tuple(purposes)
    if not {e.episode_id for e in episodes} <= {e.episode_id for e in make_split().train}:
        raise ValueError('기술 어댑터는 훈련 분할만 허용')
    if any(training != p.startswith('train:') for p in purposes):
        raise ValueError('훈련 목적과 점수 모드 불일치')
    return engine(genomes, episodes, seed, purposes, settings=NETWORK, metrics=metrics,
                  repeats=repeats, training=training, reference=reference, backend=backend)
