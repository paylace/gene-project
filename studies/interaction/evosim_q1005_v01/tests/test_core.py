import copy
import math
import random
import unittest
from unittest.mock import patch

from evosim_interaction_v02.datasets import make_split, all_episodes
from evosim_interaction_v02.environment import EpisodeSpec, LineCollectEnv
from evosim_interaction_v02.evaluation import evaluate as original_evaluate, evaluate_policy
from evosim_interaction_v02.genetics import next_population
from evosim_interaction_v02.network import initial_genome, ModuleIds
from evosim_interaction_v02.policies import CoordinatePolicy
from evosim_interaction_v02.policy import NetworkPolicy
from evosim_q1005_v01.model import NETWORK, Treatment, CONDITIONS, fitness_key, ranks_for, breed, ranking_diagnostics
from evosim_q1005_v01.evaluation import evaluate, training_metrics, replay_record, trajectory_diagnostics
from evosim_q1005_v01.audit import verify_trajectories
from evosim_q1005_v01.protocol import check_access
from evosim_q1005_v01.source import source_manifest


def record_for(episode, actions):
    env = LineCollectEnv()
    env.reset(episode)
    logged, rewards, walls, failed = [], [], 0, 0
    for action in actions:
        step = env.step(int(action))
        logged.append(str(action))
        rewards.append(round(step.reward * 100))
        walls += step.wall_collision
        failed += step.failed_collect
        if step.terminated:
            break
    return {"episode_id": episode.episode_id, "repeat": 0, "actions": "".join(logged), "rewards_cents": rewards,
            "collected": env.state.collected, "success": env.state.collected == 3, "walls": walls,
            "failed_collect": failed, "optimal_steps": episode.optimal_steps, "final_position": env.state.position}


class ProgressAndKernelTests(unittest.TestCase):
    def test_r1_total_score_uses_exact_integer_reward_units(self):
        a = {"actions": "0" * 24, "rewards_cents": [-1] * 24, "collected": 0, "success": False,
             "walls": 0, "failed_collect": 0, "optimal_steps": 10, "progress_units": [1, 1] + [0] * 22}
        b = {**a, "progress_units": [1] + [0] * 23}
        score = training_metrics([a, b], Treatment("R1S0"))["mean_training_score"]
        self.assertEqual(score, (-48 * 3 + 3 * 5) / 600)
        self.assertEqual(score, training_metrics([b, a], Treatment("R1S0"))["mean_training_score"])

    def test_progress_upper_bound_is_attainable(self):
        ep = EpisodeSpec(0, (6, 0, 6))
        record = record_for(ep, "1111112" + "0000002" + "1111112")
        units, _ = replay_record(ep, record)
        self.assertEqual(sum(units), 18)
        record["progress_units"] = units
        self.assertAlmostEqual(training_metrics([record], Treatment("R1S0"))["mean_training_score"], 3.09)

    def test_round_trip_does_not_repeat_progress(self):
        ep = EpisodeSpec(3, (0, 6, 0))
        record = record_for(ep, "0100" + "2" * 20)
        units, _ = replay_record(ep, record)
        self.assertEqual(units[:4], [1, 0, 0, 1])
        self.assertEqual(sum(units), 2)

    def test_goal_reset_collection_and_failed_collection_pay_no_bonus(self):
        ep = EpisodeSpec(0, (1, 6, 0))
        record = record_for(ep, "12" + "111112" + "0000002")
        units, _ = replay_record(ep, record)
        self.assertEqual(sum(units), 12)
        self.assertTrue(all(u == 0 for a, u in zip(record["actions"], units) if a == "2"))

    def test_wall_and_failed_collect_zero_bonus(self):
        ep = EpisodeSpec(0, (6, 0, 6))
        for actions in ("0" * 24, "2" * 24):
            units, _ = replay_record(ep, record_for(ep, actions))
            self.assertEqual(sum(units), 0)

    def test_progress_changes_tied_base_episode_scores(self):
        ep = EpisodeSpec(0, (6, 0, 6))
        rows = [record_for(ep, x) for x in ("0" * 24, "111000" + "0" * 18)]
        for row in rows:
            row["progress_units"] = replay_record(ep, row)[0]
        a, b = (training_metrics([row], Treatment("R1S0")) for row in rows)
        self.assertEqual(a["mean_reward"], b["mean_reward"])
        self.assertGreater(b["mean_training_score"], a["mean_training_score"])

    def test_r0_and_r1_actions_rewards_rng_and_genome_unchanged(self):
        episodes = make_split().train[:24]
        genome = initial_genome(random.Random(61), NETWORK)
        snapshot = genome.to_dict()
        a = evaluate(genome, episodes, 44, "train:0", Treatment("R0S0"), training=True)
        b = evaluate(genome, episodes, 44, "train:0", Treatment("R1S0"), training=True)
        self.assertEqual(a[1], b[1])
        for key in ("success_rate", "mean_reward", "mean_collected", "ties"):
            self.assertEqual(a[0][key], b[0][key])
        self.assertEqual(genome.to_dict(), snapshot)

    def test_original_r0_kernel_exact_actions_and_metrics(self):
        episodes = make_split().train[:24]
        for seed in (11, 12, 13):
            genome = initial_genome(random.Random(seed), NETWORK)
            old_metrics, old_records = original_evaluate(genome, episodes, 44, "train:0", NETWORK)
            metrics, records = evaluate(genome, episodes, 44, "train:0", Treatment("R0S0"), training=True)
            self.assertEqual(old_records, [{k: v for k, v in r.items() if k != "progress_units"} for r in records])
            self.assertTrue(all(metrics[k] == v for k, v in old_metrics.items()))
            self.assertEqual(metrics["mean_training_score"], old_metrics["mean_reward"])

    def test_scalar_batch_progress_and_reference_actions_match(self):
        episodes = make_split().train[:24]
        for seed in (21, 22, 23):
            genome = initial_genome(random.Random(seed), NETWORK)
            _, records = evaluate(genome, episodes, 44, "train:2", Treatment("R1S1"), training=True)
            _, scalar = evaluate_policy(NetworkPolicy(genome, NETWORK), episodes, 44, "train:2")
            self.assertEqual(scalar, [{k: v for k, v in row.items() if k != "progress_units"} for row in records])
            for ep, row in zip(episodes, records):
                self.assertEqual(row["progress_units"], replay_record(ep, row)[0])
            verify_trajectories(records, {ep.episode_id: ep for ep in episodes}, training=True)

    def test_validation_is_common_and_has_no_training_score(self):
        # 훈련 목록의 작은 fixture로 커널 동치만 확인하며 새 후보 held-out 성적은 열지 않는다.
        genome = initial_genome(random.Random(17), NETWORK)
        episodes = make_split().train[:12]
        outcomes = [evaluate(genome, episodes, 34, "validation", Treatment(c), repeats=3) for c in CONDITIONS]
        self.assertTrue(all(value == outcomes[0] for value in outcomes))
        self.assertNotIn("mean_training_score", outcomes[0][0])
        self.assertTrue(all("progress_units" not in r for r in outcomes[0][1]))
        self.assertEqual(outcomes[0], original_evaluate(genome, episodes, 34, "validation", NETWORK, repeats=3))

    def test_training_purpose_is_explicit(self):
        genome = initial_genome(random.Random(1), NETWORK)
        with self.assertRaises(ValueError):
            evaluate(genome, make_split().train[:1], 1, "validation", Treatment("R1S0"), training=True)
        with self.assertRaises(ValueError):
            evaluate(genome, make_split().train[:1], 1, "train:0", Treatment("R1S0"))

    def test_episode_order_and_batch_columns_are_independent(self):
        genome = initial_genome(random.Random(71), NETWORK)
        episodes = make_split().train[:5]
        _, a = evaluate(genome, episodes, 34, "train:1", Treatment("R1S0"), training=True)
        _, b = evaluate(genome, episodes[::-1], 34, "train:1", Treatment("R1S0"), training=True)
        self.assertEqual(a, b[::-1])

    def test_all_coordinate_episodes_scalar_and_batch_replay(self):
        episodes = all_episodes()
        metrics, records = evaluate_policy(CoordinatePolicy(), episodes, 71001, "coordinate_fixture")
        self.assertEqual(len(records), 1512)
        self.assertEqual(metrics["success_rate"], 1)
        lookup = {ep.episode_id: ep for ep in episodes}
        for ep, row in zip(episodes, records):
            row["progress_units"] = replay_record(ep, row)[0]
        verify_trajectories(records, lookup, training=True)

    def test_first_error_recovery_is_separate_from_immediate_accuracy(self):
        ep = EpisodeSpec(0, (6, 0, 6))
        row = record_for(ep, "0" + "1111112" + "0000002" + "1111112")
        lookup = {ep.episode_id: ep}
        a = trajectory_diagnostics([row], lookup)
        self.assertEqual(a, verify_trajectories([row], lookup))
        self.assertEqual(a["first_correct_rate"], 0)
        self.assertEqual(a["recovery_rate"], 1)

    def test_trajectory_corruption_and_progress_corruption_rejected(self):
        ep = EpisodeSpec(0, (6, 0, 6))
        row = record_for(ep, "0" * 24)
        row["progress_units"] = replay_record(ep, row)[0]
        for field in ("rewards_cents", "progress_units"):
            changed = copy.deepcopy(row)
            changed[field][0] = 1
            with self.assertRaises(ValueError):
                verify_trajectories([changed], {ep.episode_id: ep}, training=True)

    def test_data_access_guards_pilot_and_wrong_split(self):
        config = {"phase": "pilot", "validation_rounds": []}
        with self.assertRaises(ValueError):
            check_access(config, "validation", make_split().validation)
        with self.assertRaises(ValueError):
            check_access(config, "test", make_split().test)
        with self.assertRaises(ValueError):
            check_access(config, "train:1", make_split().test)
        check_access(config, "train:1", make_split().train[:24])


class FitnessTests(unittest.TestCase):
    def test_s1_completion_precedes_reward_without_weighted_sum(self):
        metrics = [{"success_rate": 1/24, "mean_training_score": -0.1}, {"success_rate": 0, "mean_training_score": 0.76}]
        self.assertEqual(ranks_for([fitness_key(m, Treatment("R0S0")) for m in metrics]), [0, 1])
        self.assertEqual(ranks_for([fitness_key(m, Treatment("R0S1")) for m in metrics]), [1, 0])

    def test_s1_equals_s0_when_all_completions_zero(self):
        metrics = [{"success_rate": 0, "mean_training_score": value} for value in (-1, -1, 0, 0.3, 0.3)]
        self.assertEqual(ranks_for([fitness_key(m, Treatment("R1S0")) for m in metrics]),
                         ranks_for([fitness_key(m, Treatment("R1S1")) for m in metrics]))

    def test_ties_and_nonfinite_keys(self):
        self.assertEqual(ranks_for([(0, 3), (0, 3), (1, -9)]), [0, 0, 1])
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                ranks_for([(value,)])

    def test_ordinal_s0_reproduction_matches_frozen_genotypes_rng_and_ancestry(self):
        rng = random.Random(123)
        a = [initial_genome(rng, NETWORK) for _ in range(6)]
        b = [g.clone() for g in a]
        ra, rb = random.Random(912), random.Random(912)
        ia, ib = ModuleIds(), ModuleIds()
        scores = [-1, -1, 0, 0.3, 0.3, 1.0]
        for _ in range(5):
            a, aa = next_population(a, scores, ra, ia, NETWORK)
            b, ab = breed(b, [(x,) for x in scores], rb, ib)
            self.assertEqual([g.to_dict() for g in a], [g.to_dict() for g in b])
            self.assertEqual(aa, ab)
            self.assertEqual(ra.getstate(), rb.getstate())
            self.assertEqual(ia.next_id, ib.next_id)

    def test_rank_diagnostics_observe_actual_conflict(self):
        rows = [{"success_rate": 0, "mean_training_score": 0.76, "mean_reward": 0.76},
                {"success_rate": 0.2, "mean_training_score": 0.384, "mean_reward": 0.384}]
        d = ranking_diagnostics(rows, Treatment("R0S1"))
        self.assertEqual(d["completion_score_conflict_pairs"], 1)
        self.assertTrue(d["s0_s1_elite_sets_changed"])
        self.assertEqual(d["applied_order"], [1, 0])

    def test_treatment_serialization_is_strict_and_immutable(self):
        for name in CONDITIONS:
            t = Treatment(name)
            self.assertEqual(Treatment.from_dict(t.to_dict()), t)
            altered = t.to_dict()
            altered["progress_unit_denominator"] = 59
            with self.assertRaises(ValueError):
                Treatment.from_dict(altered)
            with self.assertRaises(AttributeError):
                t.name = "R0S0"

    def test_frozen_dependency_manifest_is_part_of_source_guard(self):
        manifest = source_manifest()
        self.assertEqual(sum(name.startswith("dependency/") for name in manifest), 20)
        with patch("evosim_q1005_v01.source.dependency_storage.source_manifest", return_value={"changed": "x"}):
            with self.assertRaises(ValueError):
                source_manifest()


if __name__ == "__main__":
    unittest.main()
