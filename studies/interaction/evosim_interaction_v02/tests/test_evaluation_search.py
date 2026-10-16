"""W3/W4의 실제 계산·상태 격리·중단/재개·데이터 접근 검사."""

import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from evosim_interaction import evaluation as legacy_eval, network as legacy_network
from evosim_interaction.datasets import make_split as old_split
from evosim_interaction.environment import EpisodeSpec as OldEpisode
from evosim_interaction.search import run_search as legacy_search
from evosim_interaction.storage import write_json as old_write
from evosim_interaction_v02.datasets import all_episodes, make_split
from evosim_interaction_v02.evaluation import evaluate, evaluate_policy, observation_probe, select_actions
from evosim_interaction_v02.network import Genome, Module, initial_genome
from evosim_interaction_v02.policy import NetworkPolicy
from evosim_interaction_v02.policies import CoordinatePolicy
from evosim_interaction_v02.protocol import prepare, load, check_access, evaluate_split
from evosim_interaction_v02.search import run_search, directory, lock_candidates, verify_candidate_lock, attempt_accounting
from evosim_interaction_v02.settings import Settings, digest_json
from evosim_interaction_v02.storage import read_json, write_json, write_once

TEMP = Path(__file__).resolve().parents[4] / "tools/.verification_tmp"
TEMP.mkdir(exist_ok=True)


class EvaluationTests(unittest.TestCase):
    def test_a0_batch_records_and_metrics_match_reference(self):
        for seed in (201, 205, 210):
            new = initial_genome(random.Random(seed), Settings("A0"))
            old = legacy_network.Genome.from_dict(new.to_dict())
            left = evaluate(new, make_split().train[:18], 51, "validation", new.settings, repeats=3)
            right = legacy_eval.evaluate(old, old_split().train[:18], 51, "validation", repeats=3)
            self.assertEqual(left, right)

    def test_scalar_batch_records_match_for_all_conditions_and_ties(self):
        for condition in ("A0", "A1", "A2"):
            settings = Settings(condition)
            genomes = [initial_genome(random.Random(51), settings), Genome({"m0": Module("tanh", 0, 0.5)}, {}, {}, settings)]
            for genome in genomes:
                batch_metrics, records = evaluate(genome, make_split().train[:18], 15, "check", settings, repeats=2)
                scalar_metrics, scalar = evaluate_policy(NetworkPolicy(genome, settings), make_split().train[:18], 15, "check", repeats=2)
                self.assertEqual(records, scalar)
                self.assertEqual({key: batch_metrics[key] for key in scalar_metrics}, scalar_metrics)

    def test_observer_mirror_does_not_change_actions_rewards_or_genome(self):
        for condition in ("A0", "A1", "A2"):
            settings = Settings(condition)
            for tied in (False, True):
                genome = (Genome({"m0": Module("tanh", 0, 0.5)}, {}, {}, settings) if tied
                          else initial_genome(random.Random(51), settings))
                before = genome.digest
                plain = evaluate(genome, make_split().train[:8], 22, "check", settings, repeats=2)
                metrics, observed = evaluate(genome, make_split().train[:8], 22, "check", settings, repeats=2, observe=True)
                stripped = [{key: value for key, value in row.items() if key != "observations"} for row in observed]
                self.assertEqual((metrics, stripped), plain)
                self.assertEqual(genome.digest, before)
                self.assertEqual(sum(len(row["observations"]) for row in observed), sum(len(row["actions"]) for row in observed))
                for row in observed:
                    for step, item in enumerate(row["observations"]):
                        self.assertEqual(item["action"], int(row["actions"][step]))
                        self.assertEqual(item["reward_cents"], row["rewards_cents"][step])
                        self.assertEqual(item["terminated"], step == len(row["actions"]) - 1)

    def test_mirror_measures_both_correct_separately_from_switch(self):
        settings = Settings("A2")
        genome = Genome({"m0": Module("tanh", 0, 0.5)},
                        {"out_0": {"in_0": 1, "in_1": -1}, "out_1": {"in_0": -1, "in_1": 1}}, {}, settings)
        _, records = evaluate(genome, make_split().train[:12], 1, "check", settings, observe=True)
        eligible = [item["mirror"] for row in records for item in row["observations"] if item["mirror"]]
        self.assertTrue(eligible)
        self.assertTrue(all(item["both_correct"] and item["action_switched"] for item in eligible))
        bad = Genome(genome.modules, {"out_1": {"in_0": 1, "in_1": -1}, "out_0": {"in_0": -1, "in_1": 1}}, {}, settings)
        _, records = evaluate(bad, make_split().train[:12], 1, "check", settings, observe=True)
        eligible = [item["mirror"] for row in records for item in row["observations"] if item["mirror"]]
        self.assertTrue(any(item["action_switched"] and not item["both_correct"] for item in eligible))

    def test_episode_order_and_repeated_evaluation_have_independent_state(self):
        genome = initial_genome(random.Random(51), Settings("A0"))
        episodes = make_split().train[:12]
        first = evaluate(genome, episodes, 8, "check", genome.settings, repeats=2)
        reverse = evaluate(genome, episodes[::-1], 8, "check", genome.settings, repeats=2)
        keyed = lambda rows: {(r["episode_id"], r["repeat"]): r for r in rows}
        self.assertEqual(keyed(first[1]), keyed(reverse[1]))
        self.assertEqual(first, evaluate(genome, episodes, 8, "check", genome.settings, repeats=2))

    def test_probe_and_all_1512_coordinate_episodes(self):
        settings = Settings("A2")
        genome = Genome({"m0": Module("tanh", 0, 0.5)},
                        {"out_0": {"in_0": 1, "in_1": -1}, "out_1": {"in_0": -1, "in_1": 1}}, {}, settings)
        probe = observation_probe(genome, 4, settings)
        self.assertEqual(len(probe["records"]), 105)
        self.assertEqual(probe["both_correct_rate"], 1)
        metrics, records = evaluate_policy(CoordinatePolicy(), all_episodes(), 4, "foundation")
        self.assertEqual(metrics["success_rate"], 1)
        self.assertEqual(len(records), 1512)
        self.assertTrue(all(len(row["actions"]) == row["optimal_steps"] for row in records))

    def test_invalid_conditions_and_nonfinite_activity_fail(self):
        genome = initial_genome(random.Random(1), Settings("A0"))
        with self.assertRaises(ValueError):
            evaluate(genome, make_split().train[:1], 1, "check", Settings("A1"))
        with self.assertRaises(ValueError):
            evaluate(genome, (), 1, "check", genome.settings)
        # 유한 가중치라도 계산에서 overflow가 나면 기술 오류로 중단해야 한다.
        huge = Genome({"m0": Module("tanh", 0, 0.5)}, {"out_0": {"in_0": 1e308, "in_1": 1e308}}, {}, Settings("A0"))
        with self.assertRaises(FloatingPointError):
            evaluate(huge, make_split().train[-1:], 1, "check", huge.settings)


class SearchTests(unittest.TestCase):
    def temporary(self):
        return tempfile.TemporaryDirectory(dir=TEMP)

    def test_a0_five_round_search_matches_reference(self):
        with self.temporary() as temporary:
            old_root, new_root = Path(temporary) / "old", Path(temporary) / "new"
            old_write(old_root / "protocol.json", {"source_digest": "fixture"})
            prepare(new_root, "verification")
            old = legacy_search(old_root, 41, "evolution", rounds=5, size=4, pilot=True)
            new = run_search(new_root, "A0", "evolution", 41)
            for key in ("best", "initial", "final", "initial_population", "history", "training_slots", "validation_slots"):
                self.assertEqual(old[key], new[key])

    def test_pause_resume_for_each_condition_and_method(self):
        with self.temporary() as temporary:
            full, resumed = Path(temporary) / "full", Path(temporary) / "resumed"
            prepare(full, "verification")
            prepare(resumed, "verification")
            for condition in ("A0", "A1", "A2"):
                for method in ("evolution", "random"):
                    left = run_search(full, condition, method, 41)
                    run_search(resumed, condition, method, 41, pause_after=2)
                    right = run_search(resumed, condition, method, 41)
                    self.assertEqual(left, right)
                    for generation in range(5):
                        a = directory(full, condition, method, 41) / f"round_{generation:03d}.json.gz"
                        b = directory(resumed, condition, method, 41) / f"round_{generation:03d}.json.gz"
                        self.assertEqual(a.read_bytes(), b.read_bytes())
                    self.assertEqual(left, run_search(full, condition, method, 41))

    def test_shared_initial_population_and_training_batches(self):
        with self.temporary() as temporary:
            root = Path(temporary)
            prepare(root, "verification", verification_rounds=2)
            for condition in ("A0", "A1", "A2"):
                first = run_search(root, condition, "evolution", 41)
                second = run_search(root, condition, "random", 41)
                self.assertEqual(first["initial_population"], second["initial_population"])
                self.assertEqual(first["logical_episode_slots"], 4 * 2 * 24)
                self.assertEqual(attempt_accounting(directory(root, condition, "evolution", 41))["known_completed_episode_calls"], 4 * 2 * 24)
                self.assertEqual(first["validation_slots"], 0)
            batches = [read_json(directory(root, c, m, 41) / "round_000.json.gz")["training_episodes"]
                       for c in ("A0", "A1", "A2") for m in ("evolution", "random")]
            self.assertTrue(all(batch == batches[0] for batch in batches))

    def test_corrupt_or_cross_condition_checkpoint_rejected(self):
        with self.temporary() as temporary:
            root = Path(temporary)
            prepare(root, "verification")
            run_search(root, "A0", "evolution", 41, pause_after=2)
            path = directory(root, "A0", "evolution", 41) / "checkpoint.json"
            state = read_json(path)
            state["identity"]["condition"] = "A1"
            write_json(path, state)
            with self.assertRaises(ValueError):
                run_search(root, "A0", "evolution", 41)

    def test_round_or_result_tampering_rejected(self):
        with self.temporary() as temporary:
            root = Path(temporary)
            prepare(root, "verification")
            run_search(root, "A0", "evolution", 41, pause_after=2)
            path = directory(root, "A0", "evolution", 41) / "round_000.json.gz"
            write_json(path, {"corrupted": True})
            with self.assertRaises(ValueError):
                run_search(root, "A0", "evolution", 41)

    def test_pilot_verification_cannot_validate_test_or_lock(self):
        with self.temporary() as temporary:
            root = Path(temporary)
            config = prepare(root, "verification")
            for purpose, episodes in (("validation", make_split().validation), ("test", make_split().test)):
                with self.assertRaises(ValueError):
                    check_access(config, purpose, [e.episode_id for e in episodes])
                with self.assertRaises(ValueError):
                    evaluate_split(root, initial_genome(random.Random(1), Settings("A0")), episodes, 41, purpose)
            with self.assertRaises(ValueError):
                lock_candidates(root, {})
            with self.assertRaises(ValueError):
                check_access(config, "train:0", [make_split().test[0].episode_id])

    def test_protocol_source_dataset_mutation_rejected(self):
        with self.temporary() as temporary:
            root = Path(temporary)
            prepare(root, "verification")
            with patch("evosim_interaction_v02.protocol.source_manifest", return_value={"changed": "abc"}):
                with self.assertRaises(ValueError):
                    load(root)
            write_json(root / "datasets.json", {})
            with self.assertRaises(ValueError):
                load(root)

    def test_orphan_record_replayed_without_overwriting(self):
        with self.temporary() as temporary:
            root = Path(temporary)
            prepare(root, "verification", verification_rounds=2)
            from evosim_interaction_v02 import search
            original = search.write_json
            triggered = False

            def crash(path, value):
                nonlocal triggered
                if Path(path).name == "checkpoint.json" and not triggered:
                    triggered = True
                    raise OSError("simulated loss after round record")
                return original(path, value)

            with patch.object(search, "write_json", side_effect=crash), self.assertRaises(OSError):
                run_search(root, "A0", "evolution", 41)
            path = directory(root, "A0", "evolution", 41) / "round_000.json.gz"
            before = path.read_bytes()
            run_search(root, "A0", "evolution", 41)
            self.assertEqual(before, path.read_bytes())
            self.assertEqual(attempt_accounting(path.parent)["known_completed_episode_calls"], 4 * 3 * 24)
            with self.assertRaises(ValueError):
                write_once(path, {"different": True})


if __name__ == "__main__":
    unittest.main()
