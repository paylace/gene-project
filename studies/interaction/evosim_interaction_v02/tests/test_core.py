"""W1/W2의 조건 불변성·A0 정합성 검사. 본 탐색·테스트셋 평가는 실행하지 않는다."""

import copy
from dataclasses import FrozenInstanceError
import json
import random
import unittest

import numpy as np

from evosim_interaction import network as legacy
from evosim_interaction.datasets import make_split
from evosim_interaction.evaluation import evaluate_policy
from evosim_interaction.rollout import derive_seed
from evosim_interaction.search import next_population as legacy_population
from evosim_interaction_v02 import network as current
from evosim_interaction_v02.genetics import next_population
from evosim_interaction_v02.policy import BatchNetwork, NetworkPolicy
from evosim_interaction_v02.settings import Settings


class ScriptedRng:
    """금지 제안도 계획된 추첨을 정확히 소비하는지 확인하는 작은 테스트 대역."""

    def __init__(self, choices=(), randoms=(), uniforms=()):
        self.choices = list(choices)
        self.randoms = list(randoms)
        self.uniforms = list(uniforms)
        self.calls = []

    def choice(self, items):
        self.calls.append("choice")
        value = self.choices.pop(0)
        if value not in items:
            raise AssertionError((value, items))
        return value

    def random(self):
        self.calls.append("random")
        return self.randoms.pop(0)

    def uniform(self, low, high):
        self.calls.append("uniform")
        value = self.uniforms.pop(0)
        if not low <= value <= high:
            raise AssertionError(value)
        return value

    def gauss(self, mean, sigma):
        self.calls.append("gauss")
        return 0.125

    def sample(self, items, count):
        self.calls.append("sample")
        return list(items)[:count]


class CoreTests(unittest.TestCase):
    def genome(self, condition="A0", seed=31):
        settings = Settings(condition)
        return current.initial_genome(random.Random(seed), settings)

    def test_settings_deep_immutability_and_baseline_parity(self):
        settings = Settings("A0")
        self.assertEqual(settings.to_dict()["baseline_parameters"], legacy.SETTINGS)
        with self.assertRaises(FrozenInstanceError):
            settings.condition = "A2"
        with self.assertRaises(TypeError):
            settings.parameters["mutation_rate"] = 0
        with self.assertRaises(TypeError):
            settings.parameters["mutation_weights"][0] = 0
        exported = settings.to_dict()
        exported["baseline_parameters"]["mutation_weights"][0] = 0
        self.assertEqual(settings.parameters["mutation_weights"][0], 0.42)

    def test_settings_schema_and_unknown_conditions_rejected(self):
        for condition in ("A3", None, "a0"):
            with self.assertRaises(ValueError):
                Settings(condition)
        for condition in ("A0", "A1", "A2"):
            settings = Settings(condition)
            self.assertEqual(Settings.from_dict(json.loads(json.dumps(settings.to_dict()))), settings)
            data = settings.to_dict()
            data["constraints"]["module_memory"] = not data["constraints"]["module_memory"]
            with self.assertRaises(ValueError):
                Settings.from_dict(data)
        self.assertEqual(len({Settings(c).digest for c in ("A0", "A1", "A2")}), 3)
        data = Settings("A0").to_dict()
        data["constraints"]["module_memory"] = 1  # True와 값은 같아도 형식은 다름
        with self.assertRaises(ValueError):
            Settings.from_dict(data)

    def test_sixty_a0_initial_genomes_and_rng_states_match(self):
        for seed in (201, 205, 210):
            old_rng = random.Random(derive_seed("initial", seed))
            new_rng = random.Random(derive_seed("initial", seed))
            for slot in range(20):
                with self.subTest(seed=seed, slot=slot):
                    old = legacy.initial_genome(old_rng)
                    new = current.initial_genome(new_rng, Settings("A0"))
                    self.assertEqual(new.to_dict(), old.to_dict())
                    self.assertEqual(new.digest, old.digest)
                    self.assertEqual(new_rng.getstate(), old_rng.getstate())

    def test_conditions_share_prototype_and_initial_rng_consumption(self):
        for seed in (201, 205, 210):
            rngs = [random.Random(derive_seed("initial", seed)) for _ in range(3)]
            for _ in range(20):
                a0, a1, a2 = [current.initial_genome(rng, Settings(condition))
                              for rng, condition in zip(rngs, ("A0", "A1", "A2"))]
                self.assertEqual(rngs[0].getstate(), rngs[1].getstate())
                self.assertEqual(rngs[0].getstate(), rngs[2].getstate())
                self.assertEqual(a0.ff, a1.ff)
                self.assertEqual(a0.ff, a2.ff)
                self.assertEqual(a0.modules, a1.modules)
                for node, mod in a0.modules.items():
                    mapped = a2.modules[node]
                    self.assertEqual((mapped.bias, mapped.memory_decay), (mod.bias, mod.memory_decay))
                    self.assertEqual(mapped.op, mod.op if mod.op not in ("ema", "latch") else "tanh")
                expected_rec = copy.deepcopy(a0.rec)
                for out in current.OUTPUTS:
                    expected_rec[out].pop(out)
                self.assertEqual(a1.rec, expected_rec)
                self.assertEqual(a2.rec, {})

    def test_all_forced_a0_mutations_and_rng_match_legacy(self):
        settings = Settings("A0")
        for seed in (201, 205, 210):
            initial_rng = random.Random(derive_seed("initial", seed))
            for slot in range(20):
                original = legacy.initial_genome(initial_rng)
                for kind in current.KINDS:
                    old = original.clone()
                    new = current.Genome.from_dict(original.to_dict(), settings)
                    old_rng = random.Random(derive_seed("mutation_check", seed, slot, kind))
                    new_rng = random.Random(derive_seed("mutation_check", seed, slot, kind))
                    old_ids, new_ids = legacy.ModuleIds(), current.ModuleIds()
                    old_event = legacy.mutate(old, old_rng, old_ids, kind)
                    new_event = current.mutate(new, new_rng, new_ids, settings, kind)
                    with self.subTest(seed=seed, slot=slot, kind=kind):
                        self.assertEqual(new.to_dict(), old.to_dict())
                        self.assertEqual(new_rng.getstate(), old_rng.getstate())
                        self.assertEqual(new_ids.next_id, old_ids.next_id)
                        self.assertEqual({key: new_event[key] for key in old_event}, old_event)

    def test_extended_a0_mutation_crossover_and_limits_match(self):
        settings = Settings("A0")
        for seed in (201, 205, 210):
            old_rng, new_rng = random.Random(seed), random.Random(seed)
            old, new = legacy.initial_genome(old_rng), current.initial_genome(new_rng, settings)
            old_ids, new_ids = legacy.ModuleIds(), current.ModuleIds()
            for kind in ["add_module"] * 6 + ["remove_module"] * 6 + [None] * 100:
                legacy.mutate(old, old_rng, old_ids, kind)
                current.mutate(new, new_rng, new_ids, settings, kind)
                self.assertEqual(new.to_dict(), old.to_dict())
                self.assertEqual(new_rng.getstate(), old_rng.getstate())
                self.assertEqual(new_ids.next_id, old_ids.next_id)
            second_old = legacy.initial_genome(random.Random(seed + 1))
            second_new = current.Genome.from_dict(second_old.to_dict(), settings)
            for f1, f2, s1, s2 in ((0, 1, 0, 1), (1, 0, 0, 1), (1, 1, 8, 2), (1, 1, 2, 8)):
                child_old = legacy.crossover(old, second_old, old_rng, f1, f2, s1, s2)
                child_new = current.crossover(new, second_new, new_rng, f1, f2, settings, s1, s2)
                self.assertEqual(child_new.to_dict(), child_old.to_dict())
                self.assertEqual(new_rng.getstate(), old_rng.getstate())

    def test_a0_whole_population_and_ancestry_rng_match(self):
        settings = Settings("A0")
        for seed in (201, 205, 210):
            initial_rng = random.Random(derive_seed("initial", seed))
            old = [legacy.initial_genome(initial_rng) for _ in range(20)]
            new = [current.Genome.from_dict(g.to_dict(), settings) for g in old]
            old_rng = random.Random(derive_seed("search", seed, "evolution"))
            new_rng = random.Random(derive_seed("search", seed, "evolution"))
            old_ids, new_ids = legacy.ModuleIds(), current.ModuleIds()
            for generation in range(5):
                scores = [(slot % 4) / 4 for slot in range(20)]  # 동점도 포함
                old, old_records = legacy_population(old, scores, old_rng, old_ids)
                new, new_records = next_population(new, scores, new_rng, new_ids, settings)
                self.assertEqual([g.to_dict() for g in new], [g.to_dict() for g in old])
                self.assertEqual(new_rng.getstate(), old_rng.getstate())
                self.assertEqual(new_ids.next_id, old_ids.next_id)
                for left, right in zip(new_records, old_records):
                    self.assertEqual(left["kind"], right["kind"])
                    self.assertEqual(left["parents"], right["parents"])
                    if right["mutation"] is None:
                        self.assertIsNone(left["mutation"])
                    else:
                        self.assertEqual({key: left["mutation"][key] for key in right["mutation"]}, right["mutation"])

    def test_a1_only_output_self_recurrence_is_rejected(self):
        settings = Settings("A1")
        genome = self.genome("A1")
        for target, source, forbidden in (("out_0", "out_0", True), ("out_0", "out_1", False),
                                           ("m0", "m0", False), ("m0", "out_0", False)):
            child = genome.clone()
            child.rec.get(target, {}).pop(source, None)
            rng = ScriptedRng(choices=[target, source], uniforms=[0.25])
            event = current.mutate(child, rng, current.ModuleIds(), settings, "add_rec")
            self.assertEqual(rng.calls, ["choice", "choice", "uniform"])
            self.assertEqual(event["proposal_rejected"], forbidden)
            self.assertEqual(source in child.rec.get(target, {}), not forbidden)
            child.validate(settings)

    def test_a2_add_rec_is_noop_with_weight_draw_and_no_resampling(self):
        genome = self.genome("A2")
        before = genome.digest
        rng = ScriptedRng(choices=["out_0", "m0"], uniforms=[0.25])
        event = current.mutate(genome, rng, current.ModuleIds(), genome.settings, "add_rec")
        self.assertEqual(genome.digest, before)
        self.assertEqual(genome.rec, {})
        self.assertFalse(event["changed"])
        self.assertTrue(event["proposal_rejected"])
        self.assertEqual(rng.calls, ["choice", "choice", "uniform"])

    def test_duplicate_edge_does_not_draw_weight(self):
        genome = self.genome()
        before = genome.digest
        rng = ScriptedRng(choices=["out_0", "out_0"])
        event = current.mutate(genome, rng, current.ModuleIds(), genome.settings, "add_rec")
        self.assertEqual(genome.digest, before)
        self.assertEqual(rng.calls, ["choice", "choice"])
        self.assertEqual(event["rejections"][0]["reason"], "duplicate_edge")

    def test_a2_remove_rec_does_not_redirect_to_ff(self):
        genome = self.genome("A2")
        before = genome.digest
        rng = ScriptedRng(randoms=[0.1])
        event = current.mutate(genome, rng, current.ModuleIds(), genome.settings, "remove_edge")
        self.assertEqual(genome.digest, before)
        self.assertEqual(rng.calls, ["random"])
        self.assertEqual(event["rejections"][0]["reason"], "no_removable_target")

    def test_a2_module_op_rejection_keeps_bias_mutation(self):
        settings = Settings("A2")
        rng = ScriptedRng(choices=["ema"], randoms=[0, 0, 1])
        rejected = []
        result = current.Module("relu", 0.0, 0.5).mutate(rng, settings, rejected)
        self.assertEqual(result, current.Module("relu", 0.125, 0.5))
        self.assertEqual(rejected[0]["reason"], "condition_forbidden")
        self.assertEqual(rng.calls, ["random", "choice", "random", "gauss", "random"])

    def test_a1_keeps_both_module_memory_operations(self):
        for op in ("ema", "latch"):
            rng = ScriptedRng(choices=[op], randoms=[0, 1, 1])
            rejected = []
            result = current.Module("relu", 0, 0.5).mutate(rng, Settings("A1"), rejected)
            self.assertEqual(result.op, op)
            self.assertEqual(rejected, [])

    def test_a2_new_module_draws_full_prototype_before_filter(self):
        genome = self.genome("A2")
        rng = ScriptedRng(choices=["latch", "in_0", "out_1"], randoms=[0.1],
                          uniforms=[0.1, 0.7, 0.2, 0.8, 0.3])
        allocator = current.ModuleIds()
        event = current.mutate(genome, rng, allocator, genome.settings, "add_module")
        self.assertEqual(genome.modules["m4"], current.Module("tanh", 0.1, 0.7))
        self.assertEqual(genome.rec, {})
        self.assertEqual(allocator.next_id, 5)
        self.assertTrue(event["changed"])
        self.assertTrue(event["proposal_rejected"])
        self.assertEqual(len(event["rejections"]), 2)
        self.assertEqual(rng.calls, ["choice", "uniform", "uniform", "choice", "uniform",
                                     "random", "uniform", "choice", "uniform"])

    def test_weight_event_distinguishes_partial_rejection_and_change(self):
        genome = self.genome("A2")
        # 두 모듈의 op만 금지하고, 첫 모듈 bias는 변경한다.
        class WeightRng(ScriptedRng):
            def random(self):
                if self.randoms:
                    return super().random()
                self.calls.append("random")
                return 1.0  # 나머지 간선은 변경하지 않음

        rng = WeightRng(choices=["ema", "latch"], randoms=[0, 0, 1, 0, 1, 1])
        event = current.mutate(genome, rng, current.ModuleIds(), genome.settings, "weight")
        self.assertTrue(event["proposal_rejected"])
        self.assertTrue(event["changed"])
        self.assertTrue(all(mod.op not in ("ema", "latch") for mod in genome.modules.values()))

    def test_all_conditions_survive_operators_crossover_roundtrip(self):
        for condition in ("A0", "A1", "A2"):
            settings = Settings(condition)
            rng, allocator = random.Random(63), current.ModuleIds()
            first, second = current.initial_genome(rng, settings), current.initial_genome(rng, settings)
            for index in range(120):
                current.mutate(first, rng, allocator, settings, current.KINDS[index % 6])
                before = (first.digest, second.digest)
                child = current.crossover(first, second, rng, index % 2, 1 - index % 2, settings)
                self.assertEqual((first.digest, second.digest), before)
                child.validate(settings)
                child = current.Genome.restore(json.loads(json.dumps(child.export())), settings)
                child.validate(settings)
                first, second = child, first

    def test_cross_condition_inputs_fail_before_rng_or_parent_changes(self):
        first, second = self.genome("A0"), self.genome("A1")
        rng = random.Random(1)
        state, before = rng.getstate(), first.digest
        with self.assertRaises(ValueError):
            current.crossover(first, second, rng, 1, 0, first.settings)
        with self.assertRaises(ValueError):
            current.mutate(first, rng, current.ModuleIds(), second.settings)
        with self.assertRaises(ValueError):
            next_population([first, second], [1, 0], rng, current.ModuleIds(), first.settings)
        self.assertEqual(rng.getstate(), state)
        self.assertEqual(first.digest, before)

    def test_restore_rejects_condition_tampering_and_genome_corruption(self):
        genome = self.genome("A2")
        snapshot = genome.export()
        for key in ("settings_digest", "genome_digest"):
            broken = copy.deepcopy(snapshot)
            broken[key] = "wrong"
            with self.assertRaises(ValueError):
                current.Genome.restore(broken, genome.settings)
        # rec가 없는 A2 유전체라도 A1 조건의 저장본으로 가장할 수 없음.
        with self.assertRaises(ValueError):
            current.Genome.restore(snapshot, Settings("A1"))
        broken = copy.deepcopy(snapshot)
        broken["genome"]["modules"]["m0"]["bias"] += 0.1
        with self.assertRaises(ValueError):
            current.Genome.restore(broken, genome.settings)
        broken = copy.deepcopy(snapshot)
        broken["schema"] = True
        with self.assertRaises(ValueError):
            current.Genome.restore(broken, genome.settings)

    def test_invalid_or_forbidden_genomes_are_not_silently_repaired(self):
        for condition in ("A1", "A2"):
            genome = self.genome(condition)
            genome.rec["out_0"] = {"out_0": 1.0}
            with self.assertRaises(ValueError):
                current.Genome.from_dict(genome.to_dict(), genome.settings)
            self.assertEqual(genome.rec["out_0"]["out_0"], 1.0)
        genome = self.genome("A2")
        genome.rec["m0"] = {}
        with self.assertRaises(ValueError):
            genome.validate()
        genome.rec = {}
        genome.modules["m0"] = current.Module("ema", 0, 0.5)
        with self.assertRaises(ValueError):
            genome.validate()

    def test_invalid_schema_edges_cycles_numeric_values_and_ids_rejected(self):
        original = self.genome().to_dict()
        cases = []
        for value in (float("nan"), float("inf"), True, "0"):
            data = copy.deepcopy(original)
            data["ff"]["out_0"]["m0"] = value
            cases.append(data)
        dangling = copy.deepcopy(original)
        dangling["ff"]["out_0"]["m999"] = 1
        cases.append(dangling)
        cycle = copy.deepcopy(original)
        cycle["ff"]["m0"]["m1"] = 1
        cases.append(cycle)
        invalid_id = copy.deepcopy(original)
        invalid_id["modules"]["m-1"] = invalid_id["modules"].pop("m0")
        cases.append(invalid_id)
        extra = copy.deepcopy(original)
        extra["unknown"] = 1
        cases.append(extra)
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                current.Genome.from_dict(data, Settings("A0"))

    def test_module_ids_not_reused_and_dominance_uses_fitness_then_slot(self):
        settings = Settings("A0")
        rng, allocator = random.Random(8), current.ModuleIds()
        first, second = current.initial_genome(rng, settings), current.initial_genome(rng, settings)
        current.mutate(first, rng, allocator, settings, "add_module")
        current.mutate(second, rng, allocator, settings, "add_module")
        self.assertIn("m4", first.modules)
        self.assertIn("m5", second.modules)
        fitter = current.crossover(first, second, rng, 0, 1, settings)
        tied = current.crossover(first, second, rng, 1, 1, settings, 8, 2)
        self.assertIn("m5", fitter.modules)
        self.assertIn("m5", tied.modules)
        self.assertNotIn("m4", fitter.modules)
        current.mutate(first, rng, allocator, settings, "remove_module")
        current.mutate(first, rng, allocator, settings, "add_module")
        self.assertIn("m6", first.modules)

    def test_stale_allocator_and_nonfinite_scores_fail_before_changes(self):
        genome = self.genome()
        good = current.ModuleIds()
        rng = random.Random(5)
        current.mutate(genome, rng, good, genome.settings, "add_module")
        before, state = genome.digest, rng.getstate()
        stale = current.ModuleIds(4)
        with self.assertRaises(ValueError):
            current.mutate(genome, rng, stale, genome.settings, "weight")
        with self.assertRaises(ValueError):
            next_population([genome], [1], rng, stale, genome.settings)
        for value in (float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                next_population([genome], [value], rng, good, genome.settings)
        self.assertEqual(stale.next_id, 4)
        self.assertEqual(genome.digest, before)
        self.assertEqual(rng.getstate(), state)

    def test_a0_scalar_forward_action_and_rng_match(self):
        settings = Settings("A0")
        for seed in (201, 205, 210):
            rng = random.Random(derive_seed("initial", seed))
            for _ in range(20):
                old_genome = legacy.initial_genome(rng)
                new_genome = current.Genome.from_dict(old_genome.to_dict(), settings)
                old, new = legacy.NetworkPolicy(old_genome), NetworkPolicy(new_genome, settings)
                old_rng, new_rng = random.Random(seed), random.Random(seed)
                for step in range(10):
                    obs = (step / 10, 0.5, 1.0, 1 - step / 24)
                    self.assertEqual(old.forward(obs), new.forward(obs))
                    self.assertEqual(old.act(obs, old_rng), new.act(obs, new_rng))
                    self.assertEqual(old.previous, new.previous)
                    self.assertEqual(old_rng.getstate(), new_rng.getstate())

    def test_a0_batch_matches_old_backend_and_scalar_all_ops(self):
        settings = Settings("A0")
        rng = random.Random(3)
        for op in current.OPS:
            genome = current.initial_genome(rng, settings)
            genome.modules = {node: current.Module(op, mod.bias, mod.memory_decay) for node, mod in genome.modules.items()}
            old = legacy.Genome.from_dict(genome.to_dict())
            batch, old_batch = BatchNetwork(genome, 3, settings), legacy.BatchNetwork(old, 3)
            scalars = [NetworkPolicy(genome, settings) for _ in range(3)]
            for _ in range(24):
                observations = np.asarray([[rng.random() for _ in range(4)] for _ in range(3)])
                observed = batch.forward(observations)
                np.testing.assert_array_equal(observed, old_batch.forward(observations))
                expected = np.asarray([policy.forward(obs) for policy, obs in zip(scalars, observations)])
                np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)

    def test_a0_episode_records_and_reset_match_including_ties(self):
        settings = Settings("A0")
        genomes = [self.genome(seed=seed) for seed in (201, 205, 210)]
        genomes.append(current.Genome({"m0": current.Module("tanh", 0, 0.5)}, {}, {}, settings))
        for genome in genomes:
            before = genome.digest
            new_policy = NetworkPolicy(genome, settings)
            old_policy = legacy.NetworkPolicy(legacy.Genome.from_dict(genome.to_dict()))
            left = evaluate_policy(new_policy, make_split().train[:18], 33, "core_check", repeats=2)
            right = evaluate_policy(old_policy, make_split().train[:18], 33, "core_check", repeats=2)
            self.assertEqual(left, right)
            self.assertEqual(before, genome.digest)
            new_policy.reset()
            self.assertEqual(new_policy.previous, [0] * len(new_policy.nodes))
            with self.assertRaises(ValueError):
                NetworkPolicy(genome, Settings("A1"))


if __name__ == "__main__":
    unittest.main()
