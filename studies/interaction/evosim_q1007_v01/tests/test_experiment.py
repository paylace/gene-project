"""새 예산 실행기의 누출·재개·선발·통계·비용 경계를 검사한다."""
import copy
import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from evosim_interaction_v02.storage import read_json, write_json, as_tuple
from evosim_interaction_v02.settings import digest_json
from evosim_interaction_v02.datasets import make_split
from evosim_interaction_v02.rollout import derive_seed
from evosim_interaction_v02.policy import NetworkPolicy, BatchNetwork
from evosim_q1006_v02.network import initial_genome as previous_initial, mutate as previous_mutate
from evosim_q1007_v01.network import initial_genome, mutate, ModuleIds, NETWORK
from evosim_q1007_v01.protocol import fixture_config, load, check_access, shapes
from evosim_q1007_v01.search import run_search, directory, attempt_accounting
from evosim_q1007_v01.audit import audit_job
from evosim_q1007_v01.analysis import decision, paired_p, holm
from evosim_q1007_v01.finalize import lock_all, verify_lock, evaluate_locked, test_candidate
from evosim_q1007_v01.cost import gate_values


class DesignTests(unittest.TestCase):
    def test_exact_p_and_holm(self):
        self.assertEqual(paired_p(6, 0), .015625)
        self.assertEqual(paired_p(6, 0, True), .03125)
        self.assertEqual(paired_p(0, 0), 1)
        self.assertEqual(holm({'a': .02, 'b': .04}), {'a': .04, 'b': .04})

    def test_binary_not_score_sign(self):
        rows = {'B': [.1] * 80, 'PT': [.79] * 80, 'P': [.2] * 80, 'T': [.1] * 80}
        self.assertEqual(decision(rows)['eligible'], [])
        self.assertEqual(decision(rows)['contrasts']['PT-B']['ties'], 80)

    def test_success_gate_and_threshold(self):
        rows = {'B': [0.] * 80, 'PT': [1.] * 8 + [0.] * 72, 'P': [0.] * 80, 'T': [0.] * 80}
        self.assertEqual(decision(rows)['eligible'], ['PT-B'])
        rows['PT'][7] = 0.
        self.assertEqual(decision(rows)['eligible'], [])

    def test_two_sided_direction_and_inactive_family(self):
        rates = {'P': [0.] * 80, 'T': [1.] * 20 + [0.] * 60}
        row = decision(rates, active=['P-T'])
        self.assertEqual(row['eligible'], ['P-T'])
        self.assertLess(row['contrasts']['P-T']['discovery_fraction_difference'], 0)
        self.assertEqual(row['contrasts']['P-T']['holm_p'], row['contrasts']['P-T']['raw_p'] * 2)

    def test_malformed_sample_rejected(self):
        with self.assertRaises(ValueError):
            decision({'P': [0.], 'T': [0.]}, active=['P-T'])

    def test_parameter_count_and_initial_equivalence(self):
        a = initial_genome(random.Random(14))
        b = previous_initial(random.Random(14))
        self.assertEqual(a.to_dict(), b.to_dict())
        self.assertEqual(sum(len(e) for e in a.ff.values()) + len(a.modules), 32)

    def test_mutation_preserves_previous_rng(self):
        a, b = (initial_genome(random.Random(14)) for _ in range(2))
        ra, rb = random.Random(16), random.Random(16)
        for kind in ('weight', 'add_module', 'remove_module', 'add_ff', 'add_rec', 'remove_edge'):
            self.assertEqual(mutate(a, ra, ModuleIds(), 'F0', kind), previous_mutate(b, rb, ModuleIds(), 'F0', kind))
            self.assertEqual(a.digest, b.digest)
            self.assertEqual(ra.getstate(), rb.getstate())

    def test_batch_scalar_and_freeze(self):
        genome = initial_genome(random.Random(29))
        before = genome.digest
        scalar = NetworkPolicy(genome, NETWORK)
        batch = BatchNetwork(genome, 3, NETWORK)
        obs = np.asarray([[0., .5, 1., 1.], [1., 0., .6, .5], [.2, .2, .3, .1]])
        values = batch.forward(obs)
        expected = [scalar.forward(row) for row in obs]
        self.assertTrue(np.allclose(values, expected, rtol=1e-12, atol=1e-12))
        self.assertEqual(before, genome.digest)

    def test_population_prefix(self):
        a = random.Random(derive_seed('q1007:v01:initial', 2001))
        b = random.Random(derive_seed('q1007:v01:initial', 2001))
        twenty = [initial_genome(a).digest for _ in range(20)]
        forty = [initial_genome(b).digest for _ in range(40)]
        self.assertEqual(twenty, forty[:20])

    def test_validation_schedule_and_endpoint_shapes(self):
        self.assertEqual(shapes('main')['E20']['endpoint_rounds'], {'100': 'B', '200': 'T'})
        self.assertEqual(shapes('main')['R20']['rounds'], 400)
        self.assertEqual(shapes('pilot')['E40']['rounds'], 20)

    def test_cost_boundaries(self):
        self.assertEqual(gate_values(5000, 2200, 20 * 2**30, 45 * 2**30), [])
        self.assertTrue(gate_values(5000, 2201, 20 * 2**30 + 1, 0))


class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_root = Path(__file__).resolve().parents[4] / 'tools/.verification_tmp'
        cls.temporary_root.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=cls.temporary_root)
        cls.root = Path(cls.temp.name)
        cls.full, cls.resumed, cls.short = (cls.root / name for name in ('full', 'resumed', 'short'))
        for root in (cls.full, cls.resumed):
            fixture_config(root, rounds=20, endpoints={'10': 'early', '20': 'late'})
        fixture_config(cls.short, rounds=10, endpoints={'10': 'early'})
        cls.full_result = run_search(cls.full, 'F', 'evolution', 41)
        run_search(cls.resumed, 'F', 'evolution', 41, pause_after=7)
        cls.resumed_result = run_search(cls.resumed, 'F', 'evolution', 41)
        cls.short_result = run_search(cls.short, 'F', 'evolution', 41)

    @classmethod
    def tearDownClass(cls):
        if not cls.root.resolve().is_relative_to(cls.temporary_root.resolve()):
            raise ValueError('검증 임시 경로가 작업 공간을 벗어남')
        cls.temp.cleanup()

    def test_resume_identical(self):
        self.assertEqual(self.full_result, self.resumed_result)

    def test_prefix_endpoint_and_population(self):
        self.assertEqual(self.short_result['endpoints']['early'], self.full_result['endpoints']['early'])
        for g in range(10):
            left = read_json(directory(self.full, 'F', 'evolution', 41) / f'round_{g:03d}.json.gz')
            right = read_json(directory(self.short, 'F', 'evolution', 41) / f'round_{g:03d}.json.gz')
            self.assertEqual(left, right)

    def test_audit_and_accounting(self):
        audit = audit_job(self.resumed, 'F', 'evolution', 41)
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['episode_records_replayed'], 2 * (20 * 24 + 4 * 2 * 3))
        row = attempt_accounting(directory(self.resumed, 'F', 'evolution', 41))
        self.assertEqual(row['interrupted_attempts'], 0)
        self.assertGreater(row['known_single_step_actions'], 0)

    def test_only_validation_candidates(self):
        for item in self.full_result['endpoints'].values():
            self.assertEqual(item['candidate']['selection_source'], 'validation')
            self.assertLess(item['candidate']['round'], item['through_round'])

    def test_no_initial_validation(self):
        config = load(self.full, 'F')
        self.assertNotIn(0, config['validation_rounds'])
        self.assertIsNone(self.full_result['initial'])

    def test_test_partition_access_and_lock_rejected(self):
        with self.assertRaises(ValueError):
            check_access(load(self.full, 'F'), 'train:0', make_split().test[:1])
        with self.assertRaises(ValueError):
            lock_all(self.full)
        with self.assertRaises((ValueError, FileNotFoundError)):
            evaluate_locked(self.full)

    def test_tamper_checkpoint_rejected(self):
        path = self.root / 'tamper'
        fixture_config(path, rounds=1, validation=False)
        run_search(path, 'F', 'evolution', 41)
        checkpoint = directory(path, 'F', 'evolution', 41) / 'checkpoint.json'
        data = read_json(checkpoint)
        data['next_round'] = 0
        write_json(checkpoint, data)
        with self.assertRaises(ValueError):
            run_search(path, 'F', 'evolution', 41)

    def test_test_cache_freeze_and_wrong_key(self):
        from evosim_q1007_v01.evaluation import evaluate
        config = load(self.full)
        locked = {'lock_digest': 'synthetic-lock'}
        candidate = self.full_result['endpoints']['late']['candidate']
        split = make_split()
        # 본 테스트를 열지 않고, 훈련 분할504개로 평가·캐시의 실제 경로를 검사한다.
        synthetic = type('FixtureSplit', (), {'test': split.train[:504]})()
        with patch('evosim_q1007_v01.finalize.make_split', return_value=synthetic), patch('evosim_q1007_v01.finalize.evaluate', wraps=evaluate) as evaluator:
            first, path = test_candidate(self.full, config, locked, candidate, 41)
            second, _ = test_candidate(self.full, config, locked, candidate, 41)
            self.assertEqual(evaluator.call_count, 1)
            self.assertEqual(first, second)
            self.assertEqual(len(first['records']), 1512)
            with self.assertRaises(ValueError):
                test_candidate(self.full, config, {'lock_digest': 'changed'}, candidate, 41)

    def test_round_tamper_audit_rejected(self):
        path = self.root / 'round_tamper'
        fixture_config(path, rounds=1, validation=False)
        run_search(path, 'F', 'evolution', 41)
        record = directory(path, 'F', 'evolution', 41) / 'round_000.json.gz'
        data = read_json(record)
        data['candidates'][0]['training_records'][0]['actions'] = '2'
        write_json(record, data)
        with self.assertRaises(ValueError):
            audit_job(path, 'F', 'evolution', 41)


if __name__ == '__main__':
    unittest.main()
