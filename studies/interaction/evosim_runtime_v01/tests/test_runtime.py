"""허용 오차로 숨기지 않는 행동·기록 동치와 증거 보존·원자 저장 검사."""
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from evosim_interaction_v02.datasets import make_split
from evosim_interaction_v02.policy import BatchNetwork
from evosim_q1007_v01.network import initial_genome, NETWORK, Module
from evosim_q1007_v01.audit import verify_trajectories
from evosim_q1007_v02.evaluation import evaluate_population
from evosim_q1007_v02.fixture import run_fixture
from evosim_runtime_v01.linecollect import FixedTanhBackend
from evosim_runtime_v01.records import LineCollectRecords
from evosim_runtime_v01.storage import TableCodec, Store, canonical, digest
from evosim_runtime_v01.accounting import COUNTS

ROOT = Path(__file__).resolve().parents[4]


def genomes(n=4):
    rng = random.Random(3101)
    return [initial_genome(rng) for _ in range(n)]


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent = ROOT / 'tools/.verification_tmp'
        parent.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=parent)
        cls.root = Path(cls.temp.name)
        cls.episodes = make_split().train[:6]

    @classmethod
    def tearDownClass(cls):
        if not cls.root.resolve().is_relative_to((ROOT / 'tools/.verification_tmp').resolve()):
            raise ValueError('검사 임시 경로가 작업 공간 밖임')
        cls.temp.cleanup()

    def compare(self, population, *, repeats=1, training=False):
        purpose = 'train:fixture' if training else 'fixture'
        purposes = [purpose + f':{i}' for i in range(len(population))]
        kwargs = dict(repeats=repeats, training=training)
        baseline = evaluate_population(population, self.episodes, 3101, purposes, backend='reference', **kwargs)
        optimized = evaluate_population(population, self.episodes, 3101, purposes, backend='fixed', **kwargs)
        self.assertEqual(canonical(baseline), canonical(optimized))
        return optimized

    def test_forward_exact(self):
        population = genomes(8)
        rng = np.random.default_rng(3101)
        obs = rng.random((8, 19, 4))
        actual = FixedTanhBackend(population).forward(obs)
        for i, g in enumerate(population):
            self.assertTrue(np.array_equal(actual[i], BatchNetwork(g, 19, NETWORK).forward(obs[i])))

    def test_random_training_records_and_audit(self):
        result = self.compare(genomes(), repeats=2, training=True)
        lookup = {ep.episode_id: ep for ep in self.episodes}
        for _, records in result:
            verify_trajectories(records, lookup, training=True)

    def test_validation_and_frozen(self):
        population = genomes()
        before = canonical([g.to_dict() for g in population])
        self.compare(population, repeats=3)
        self.assertEqual(before, canonical([g.to_dict() for g in population]))

    def test_exact_ties_and_near_ties(self):
        population = genomes(3)
        for g in population:
            for edges in g.ff.values():
                for source in edges:
                    edges[source] = 0.
            for name in g.modules:
                g.modules[name] = Module('tanh', 0., .5)
        population[1].ff['out_1']['m0'] = 1e-10
        population[2].ff['out_2']['m0'] = -1e-10
        self.compare(population, repeats=3, training=True)

    def test_boundary_weights(self):
        population = genomes(2)
        for index, g in enumerate(population):
            for name in g.modules:
                g.modules[name] = Module('tanh', (-1 if index else 1) * 2., .5)
            for edges in g.ff.values():
                for source in edges:
                    edges[source] = (-1 if index else 1) * 3.
        self.compare(population, training=True)

    def test_early_success_goal_changes_and_no_unneeded_rng(self):
        g = genomes(1)[0]
        for name in g.modules:
            g.modules[name] = Module('tanh', .1 if name == 'm2' else 0., .5)
        for edges in g.ff.values():
            for source in edges:
                edges[source] = 0.
        g.ff['m0'].update(in_0=2., in_1=-2.)
        g.ff['m1'].update(in_0=-2., in_1=2.)
        for i in range(3):
            g.ff[f'out_{i}'] = {f'm{j}': float(i == j) for j in range(4)}
        before = COUNTS['rng_objects_created']
        result = self.compare([g], repeats=2, training=True)
        self.assertEqual(COUNTS['rng_objects_created'], before)
        self.assertEqual(result[0][0]['success_rate'], 1.)
        self.assertTrue(all(r['success'] and len(r['actions']) == r['optimal_steps'] for r in result[0][1]))

    def test_heldout_access_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_population(genomes(1), make_split().test[:1], 3101, ['fixture'])

    def test_other_topology_fallback(self):
        population = genomes(1)
        del population[0].ff['m0']['in_0']
        purposes = ['fixture:fallback']
        baseline = evaluate_population(population, self.episodes, 3101, purposes, backend='reference')
        actual = evaluate_population(population, self.episodes, 3101, purposes)
        self.assertEqual(canonical(baseline), canonical(actual))
        with self.assertRaises(ValueError):
            evaluate_population(population, self.episodes, 3101, purposes, backend='fixed')

    def test_candidate_independence(self):
        population = genomes(3)
        purposes = [f'fixture:{i}' for i in range(3)]
        together = evaluate_population(population, self.episodes, 3101, purposes)
        for g, purpose, expected in zip(population, purposes, together):
            self.assertEqual(canonical(expected), canonical(evaluate_population([g], self.episodes, 3101, [purpose])[0]))

    def sample(self):
        result = evaluate_population(genomes(1), self.episodes, 3101, ['train:codec'], training=True)[0]
        return dict(genome=genomes(1)[0].to_dict(), metrics=result[0], training_records=result[1])

    def test_codec_preserves_all_fields_and_errors(self):
        value = self.sample()
        value['training_records'][0]['rewards_cents'][0] = 12345
        value['training_records'][1]['new_future_field'] = {'a': [None, True, 1.0]}
        value['training_records'][2]['success'] = 0
        del value['training_records'][3]['walls']
        codec = TableCodec(LineCollectRecords())
        restored = codec.decode(codec.encode(value))
        self.assertEqual(canonical(restored), canonical(value))

    def test_invalid_original_trajectory_preserved_literal(self):
        value = self.sample()
        value['training_records'][0]['actions'] = '9'
        codec = TableCodec(LineCollectRecords())
        self.assertEqual(canonical(codec.decode(codec.encode(value))), canonical(value))

    def test_codec_tamper_and_wrong_environment(self):
        codec = TableCodec(LineCollectRecords())
        encoded = codec.encode(self.sample())
        broken = copy.deepcopy(encoded)
        broken['tables']['actions'][0] = '2'
        with self.assertRaises(ValueError):
            codec.decode(broken)
        broken['content_sha256'] = digest({k: v for k, v in broken.items() if k != 'content_sha256'})
        with self.assertRaises((ValueError, RuntimeError)):
            codec.decode(broken)
        wrong = copy.deepcopy(encoded)
        wrong['dependencies']['environment_sha256'] = 'changed'
        wrong['content_sha256'] = digest({k: v for k, v in wrong.items() if k != 'content_sha256'})
        with self.assertRaises(ValueError):
            codec.decode(wrong)

    def test_store_plain_json_and_codec_and_immutable(self):
        value = dict(ordinary={'records': 'not an environment record', 'unknown': [1, True]})
        for index, store in enumerate((Store(), Store(TableCodec()))):
            path = self.root / f'plain_{index}.json.gz'
            store.write(path, value)
            self.assertEqual(store.read(path), value)
            self.assertFalse(store.write(path, value))
            with self.assertRaises(ValueError):
                store.write(path, {})

    def test_failed_atomic_replace_keeps_original(self):
        store = Store()
        path = self.root / 'atomic.json.gz'
        store.write(path, {'before': 1})
        with patch('evosim_runtime_v01.storage.os.replace', side_effect=OSError('synthetic interrupted commit')):
            with self.assertRaises(OSError):
                store.write(path, {'after': 2}, replace=True)
        self.assertEqual(store.read(path), {'before': 1})
        self.assertFalse(list(path.parent.glob(path.name + '.*.tmp')))

    def test_fixture_resume_and_reference_equal(self):
        full = run_fixture(self.root / 'full')
        run_fixture(self.root / 'resume', pause_after=4)
        resumed = run_fixture(self.root / 'resume')
        baseline = run_fixture(self.root / 'reference', backend='reference')
        self.assertEqual(canonical(full), canonical(resumed))
        self.assertEqual(canonical(full), canonical(baseline))

    def test_fixture_rejects_changed_round(self):
        root = self.root / 'tampered'
        run_fixture(root, rounds=1)
        path = root / 'round_000.json.gz'
        store = Store(TableCodec(LineCollectRecords()))
        data = store.read(path)
        data['round'] = 8
        store.write(path, data, replace=True)
        with self.assertRaises(ValueError):
            run_fixture(root, rounds=1)

    def test_wrong_training_mode_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_population(genomes(1), self.episodes, 3101, ['test'], training=True)


if __name__ == '__main__':
    unittest.main()
