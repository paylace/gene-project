"""측정된 평가·저장 프로필을 한곳에서 조합하는 기술 실행 API."""
import time
from evosim_q1007_v02.evaluation import evaluate_population
from evosim_runtime_v01.storage import Store, TableCodec as ReplayCodec, canonical
from evosim_runtime_v01.records import LineCollectRecords
from evosim_runtime_v02.storage import TableCodec as ObservedCodec

PROFILES = ('reference', 'balanced', 'archive')


class ExperimentRuntime:
    def __init__(self, profile='balanced'):
        if profile not in PROFILES:
            raise ValueError('알 수 없는 실행 프로필')
        self.profile = profile
        self.store = (Store(level=3) if profile == 'reference' else
                      Store(ObservedCodec(), level=9) if profile == 'balanced' else
                      Store(ReplayCodec(LineCollectRecords()), level=3))

    def evaluate_and_store(self, path, genomes, episodes, seed, purposes, *, training=False, repeats=1):
        """훈련 전용. 실제 평가→보존→복원 검증의 시간을 각각 반환한다."""
        genomes = tuple(genomes)
        start = time.perf_counter()
        outcomes = evaluate_population(genomes, episodes, seed, purposes, training=training,
            repeats=repeats, backend='reference' if self.profile == 'reference' else 'auto')
        evaluated = time.perf_counter()
        field = 'training_records' if training else 'validation_records'
        payload = {'candidates': [dict(slot=i, genome=g.to_dict(), metrics=metrics, **{field: records})
                                  for i, (g, (metrics, records)) in enumerate(zip(genomes, outcomes))]}
        self.store.write(path, payload)
        saved = time.perf_counter()
        restored = self.store.read(path)
        if canonical(restored) != canonical(payload):
            raise AssertionError('평가 후 보존한 전체 기록이 다름')
        checked = time.perf_counter()
        return payload, dict(evaluation_seconds=evaluated - start, storage_seconds=saved - evaluated,
                             restore_and_check_seconds=checked - saved, total_seconds=checked - start)
