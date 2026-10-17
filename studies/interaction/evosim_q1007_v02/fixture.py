"""훈련 전용 짧은 탐색으로 평가·저장·중단/재개를 함께 확인한다. 본 진입점 아님."""
import random
import uuid
from pathlib import Path
from evosim_interaction_v02.datasets import make_split
from evosim_interaction_v02.rollout import derive_seed
from evosim_interaction_v02.storage import as_tuple
from evosim_q1007_v01.network import initial_genome, Genome, ModuleIds, NETWORK
from evosim_q1007_v01.model import breed, fitness_key, Treatment
from evosim_q1007_v01.evaluation import selection_key
from evosim_runtime_v01.storage import Store, TableCodec, digest
from evosim_runtime_v01.records import LineCollectRecords
from .evaluation import evaluate_population
from .provenance import source_manifest


def run_fixture(root, *, seed=3102, population=4, rounds=10, pause_after=None,
                backend='auto', validation=True):
    if type(population) is not int or not 2 <= population <= 40 or not 1 <= rounds <= 20:
        raise ValueError('기술 fixture 크기 범위 초과')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    store = Store(TableCodec(LineCollectRecords()))
    identity = dict(seed=seed, population=population, rounds=rounds, validation=validation,
                    rng_namespace='q1007:v01', backend=backend, technical_only=True,
                    source_digest=digest(source_manifest()), datasets_digest=make_split().digest,
                    settings_digest=NETWORK.digest)
    store.write(root / 'protocol.json.gz', identity)
    path = root / 'checkpoint.json.gz'
    if path.exists():
        state = store.read(path)
        if state['identity'] != identity:
            raise ValueError('재개 식별자 변경')
        if state['state_digest'] != digest({k: v for k, v in state.items() if k != 'state_digest'}):
            raise ValueError('체크포인트 내용 변경')
        if set(state['hashes']) != {f'round_{r:03d}.json.gz' for r in range(state['next_round'])}:
            raise ValueError('완료 회차 목록 누락')
        import hashlib
        if any(hashlib.sha256((root / n).read_bytes()).hexdigest() != d for n, d in state['hashes'].items()):
            raise ValueError('완료 회차 변경')
        if state['next_round'] == rounds:
            return store.read(root / 'result.json.gz')
        genomes = [Genome.from_dict(g, NETWORK) for g in state['population']]
        rng = random.Random()
        rng.setstate(as_tuple(state['rng_state']))
        allocator = ModuleIds(state['next_module_id'])
    else:
        rng_init = random.Random(derive_seed('q1007:v01:initial', seed))
        genomes = [initial_genome(rng_init) for _ in range(population)]
        rng = random.Random(derive_seed('q1007:v01:search', seed, 'evolution'))
        allocator = ModuleIds()
        state = dict(identity=identity, next_round=0, best=None, hashes={}, history=[], calls=0,
                     ancestry=[{'kind': 'initial', 'parents': [], 'mutation': None} for _ in genomes])
    split = make_split()
    for generation in range(state['next_round'], rounds):
        attempt_path = root / 'attempts' / f'{generation:03d}_{uuid.uuid4().hex}.json.gz'
        attempt = dict(round=generation, status='started', completed_episode_calls=0,
                       maximum_episode_calls=population * (24 + (6 if validation and (generation + 1) % 5 == 0 else 0)))
        store.write(attempt_path, attempt)
        episodes = random.Random(derive_seed('q1007:v01:training_batch', seed, generation)).sample(list(split.train), 24)
        purposes = [f'train:{generation}:q1007:v01:slot{i}' for i in range(population)]
        outcomes = evaluate_population(genomes, episodes, seed, purposes, training=True, backend=backend)
        attempt.update(status='training_evaluated', completed_episode_calls=population * 24)
        store.write(attempt_path, attempt, replace=True)
        candidates, keys = [], []
        for slot, (genome, (metrics, records)) in enumerate(zip(genomes, outcomes)):
            keys.append(fitness_key(metrics, Treatment('F0')))
            candidates.append(dict(slot=slot, genome=genome.to_dict(), ancestry=state['ancestry'][slot],
                                   metrics=metrics, training_records=records))
        state['calls'] += population * 24
        if validation and (generation + 1) % 5 == 0:
            # 실제 validation/test를 열지 않는 훈련 분할의 기술 표본.
            outcomes = evaluate_population(genomes, split.train[:2], seed,
                [f'validation:q1007:v01:slot{i}' for i in range(population)], repeats=3, backend=backend)
            state['calls'] += population * 6
            for row, (metrics, records) in zip(candidates, outcomes):
                row.update(validation=metrics, validation_records=records)
                candidate = dict(genome=row['genome'], metrics=metrics, round=generation, slot=row['slot'])
                if state['best'] is None or selection_key(metrics) > selection_key(state['best']['metrics']):
                    state['best'] = candidate
        attempt.update(status='evaluated', completed_episode_calls=population * (24 + (6 if validation and (generation + 1) % 5 == 0 else 0)))
        store.write(attempt_path, attempt, replace=True)
        name = f'round_{generation:03d}.json.gz'
        store.write(root / name, dict(round=generation, candidates=candidates))
        import hashlib
        state['hashes'][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
        state['history'].append(dict(round=generation, keys=keys, population=[g.digest for g in genomes], best=state['best']))
        if generation < rounds - 1:
            genomes, state['ancestry'] = breed(genomes, keys, rng, allocator, 'F0')
        state.update(next_round=generation + 1, population=[g.to_dict() for g in genomes],
                     rng_state=rng.getstate(), next_module_id=allocator.next_id)
        result = dict(history=state['history'], best=state['best'], calls=state['calls'],
                      final_population=state['population'], rng_state=state['rng_state'])
        if generation == rounds - 1:
            store.write(root / 'result.json.gz', result)
        state['state_digest'] = digest({k: v for k, v in state.items() if k != 'state_digest'})
        store.write(path, state, replace=True)
        attempt['status'] = 'committed'
        store.write(attempt_path, attempt, replace=True)
        if pause_after is not None and generation + 1 >= pause_after:
            return dict(paused_after=generation + 1)
    return result
