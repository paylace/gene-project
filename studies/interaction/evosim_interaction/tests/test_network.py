import json
import random
import subprocess
import sys
import unittest

import numpy as np

from evosim_interaction.datasets import make_split
from evosim_interaction.evaluation import evaluate, evaluate_policy, observation_probe
from evosim_interaction.network import (
    BatchNetwork, Genome, Module, ModuleIds, NetworkPolicy, crossover, initial_genome, mutate,
)


class NetworkTests(unittest.TestCase):
    def test_fourth_input_reaches_output_and_bad_dimension_rejected(self):
        genome = Genome({"m0": Module("tanh", 0, 0.5)},
                        {"m0": {"in_3": 1.0}, "out_0": {"m0": 1.0}}, {})
        policy = NetworkPolicy(genome)
        self.assertEqual(policy.forward((0, 0, 0, 0))[0], 0)
        self.assertGreater(policy.forward((0, 0, 0, 1))[0], 0.7)
        with self.assertRaises(ValueError):
            policy.forward((0, 0, 0))

    def test_temporal_state_and_snapshot_parameters_are_separate(self):
        genome = initial_genome(random.Random(1))
        original = genome.digest
        policy = NetworkPolicy(genome)
        first = policy.forward((0.2, 0.5, 1, 1))
        policy.forward((0.2, 0.5, 1, 1))
        policy.reset()
        self.assertEqual(first, policy.forward((0.2, 0.5, 1, 1)))
        self.assertEqual(genome.digest, original)
        self.assertEqual(Genome.from_dict(json.loads(json.dumps(genome.to_dict()))).digest, original)
        self.assertFalse(hasattr(policy, "value_estimates"))
        self.assertFalse(hasattr(policy, "experience_buffer"))

    def test_module_ids_are_shared_by_run_never_reused_across_lineages(self):
        rng = random.Random(8)
        allocator = ModuleIds()
        first, second = initial_genome(rng), initial_genome(rng)
        mutate(first, rng, allocator, "add_module")
        mutate(second, rng, allocator, "add_module")
        self.assertIn("m4", first.modules)
        self.assertIn("m5", second.modules)
        mutate(first, rng, allocator, "remove_module")
        mutate(first, rng, allocator, "add_module")
        self.assertIn("m6", first.modules)
        self.assertEqual(allocator.next_id, 7)

    def test_mutation_and_crossover_keep_valid_graphs_and_fitter_unique_module(self):
        rng = random.Random(44)
        allocator = ModuleIds()
        first, second = initial_genome(rng), initial_genome(rng)
        mutate(second, rng, allocator, "add_module")
        before_first, before_second = first.digest, second.digest
        child = crossover(first, second, rng, 0, 1)
        self.assertIn("m4", child.modules)
        self.assertEqual(first.digest, before_first)
        self.assertEqual(second.digest, before_second)
        for _ in range(100):
            mutate(child, rng, allocator)
            child = crossover(child, first, rng, 1, 0)
            child.validate()
        invalid = Genome({"m0": Module("tanh", 0, 0.5), "m1": Module("relu", 0, 0.5)},
                         {"m0": {"m1": 1}, "m1": {"m0": 1}}, {})
        with self.assertRaises(ValueError):
            invalid.validate()

    def test_numpy_matches_scalar_network_for_all_module_operations(self):
        rng = random.Random(3)
        for op in ("linear", "threshold", "relu", "tanh", "ema", "latch"):
            genome = initial_genome(rng)
            genome.modules = {key: Module(op, mod.bias, mod.memory_decay) for key, mod in genome.modules.items()}
            batch = BatchNetwork(genome, 3)
            scalar = [NetworkPolicy(genome) for _ in range(3)]
            for _ in range(24):
                observations = np.array([[rng.random() for _ in range(4)] for _ in range(3)])
                expected = np.array([policy.forward(obs) for policy, obs in zip(scalar, observations)])
                np.testing.assert_allclose(batch.forward(observations), expected, rtol=1e-12, atol=1e-12)

    def test_batched_episode_actions_rewards_and_resets_match_scalar(self):
        episodes = make_split().train[:18]
        rng = random.Random(52)
        genomes = [initial_genome(rng) for _ in range(4)]
        # Exactly tied outputs exercise the seeded tie-breaking path.
        genomes.append(Genome({"m0": Module("tanh", 0, 0.5)}, {}, {}))
        for genome in genomes:
            before = genome.digest
            metrics, records = evaluate(genome, episodes, 18, "check", repeats=2)
            scalar_metrics, scalar_records = evaluate_policy(NetworkPolicy(genome), episodes, 18, "check", repeats=2)
            self.assertEqual(records, scalar_records)
            for key, value in scalar_metrics.items():
                self.assertEqual(metrics[key], value)
            self.assertEqual(genome.digest, before)
            reverse_metrics, reverse = evaluate(genome, episodes[::-1], 18, "check", repeats=2)
            keyed = lambda rows: {(row["episode_id"], row["repeat"]): row for row in rows}
            self.assertEqual(keyed(records), keyed(reverse))

    def test_observation_probe_detects_direction_use_and_blind_failure(self):
        genome = Genome({"m0": Module("tanh", 0, 0.5)},
                        {"out_0": {"in_0": 1, "in_1": -1},
                         "out_1": {"in_0": -1, "in_1": 1}}, {})
        result = observation_probe(genome, 2)
        self.assertEqual(len(result["records"]), 105)
        self.assertEqual(result["both_correct_rate"], 1)
        blind = Genome({"m0": Module("tanh", 0, 0.5)}, {}, {})
        self.assertEqual(observation_probe(blind, 2)["both_correct_rate"], 0)

    def test_cross_process_seed_and_genome_reproducibility(self):
        code = ("import random; from evosim_interaction.network import initial_genome; "
                "from evosim_interaction.datasets import make_split; "
                "print(initial_genome(random.Random(4)).digest,make_split().digest)")
        first = subprocess.check_output([sys.executable, "-c", code], text=True)
        second = subprocess.check_output([sys.executable, "-c", code], text=True)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
