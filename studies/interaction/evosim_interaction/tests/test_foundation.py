"""성능 실험에 앞서 환경·분할·기준 정책의 계약을 검증한다."""

import math
import random
import unittest
from collections import Counter

from evosim_interaction.datasets import all_episodes, make_split, reflect
from evosim_interaction.environment import (
    ACTION_BUDGET, COLLECT, LEFT, RIGHT, EpisodeSpec, LineCollectEnv,
)
from evosim_interaction.policies import (
    CoordinatePolicy, FixedSequencePolicy, UniformRandomPolicy, blind_patterns,
)
from evosim_interaction.rollout import derive_seed, run_episode


class EnvironmentTests(unittest.TestCase):
    def test_coordinate_policy_solves_every_case_optimally(self):
        episodes = all_episodes()
        self.assertEqual(len(episodes), 1512)
        records = [run_episode(CoordinatePolicy(), episode, 1) for episode in episodes]
        self.assertEqual(max(record.steps for record in records), 21)
        for record in records:
            with self.subTest(episode=record.episode.episode_id):
                self.assertTrue(record.success)
                self.assertEqual(record.steps, record.episode.optimal_steps)
                self.assertAlmostEqual(record.total_reward, 3 - 0.01 * record.steps)
                self.assertEqual(sum(t.collected for t in record.transitions), 3)
                self.assertFalse(any(t.wall_collision or t.failed_collect for t in record.transitions))

    def test_hand_calculated_example(self):
        record = run_episode(CoordinatePolicy(), EpisodeSpec(2, (4, 1, 5)), 1)
        self.assertEqual(record.steps, 12)
        self.assertAlmostEqual(record.total_reward, 2.88)
        transitions = record.transitions
        self.assertEqual(transitions[2].action, COLLECT)
        self.assertEqual(transitions[3].before.goal, 1)
        self.assertEqual(transitions[3].action, LEFT)
        self.assertEqual(transitions[7].before.goal, 5)
        self.assertEqual(transitions[7].action, RIGHT)

    def test_failed_collect_and_wall_consume_budget_without_reward(self):
        env = LineCollectEnv()
        env.reset(EpisodeSpec(0, (2, 0, 6)))
        wall = env.step(LEFT)
        self.assertTrue(wall.wall_collision)
        self.assertEqual(env.state.position, 0)
        failed = env.step(COLLECT)
        self.assertTrue(failed.failed_collect)
        self.assertEqual(wall.reward, -0.01)
        self.assertEqual(failed.reward, -0.01)
        self.assertEqual(env.state.budget, 22)
        self.assertEqual(env.state.goal, 2)
        self.assertEqual(env.state.collected, 0)
        env.reset(EpisodeSpec(6, (2, 0, 6)))
        self.assertTrue(env.step(RIGHT).wall_collision)

    def test_arrival_requires_collect_and_each_collect_advances_once(self):
        env = LineCollectEnv()
        env.reset(EpisodeSpec(0, (1, 0, 1)))
        result = env.step(RIGHT)
        self.assertFalse(result.collected)
        self.assertEqual(env.state.collected, 0)
        collected = env.step(COLLECT)
        self.assertEqual(collected.reward, 0.99)
        self.assertEqual(env.state.goal, 0)
        repeated = env.step(COLLECT)
        self.assertTrue(repeated.failed_collect)
        self.assertEqual(env.state.collected, 1)

    def test_third_collect_on_last_action_is_success(self):
        pattern = (LEFT,) * 18 + (RIGHT, COLLECT, LEFT, COLLECT, RIGHT, COLLECT)
        record = run_episode(FixedSequencePolicy(pattern), EpisodeSpec(0, (1, 0, 1)), 1)
        self.assertEqual(record.steps, ACTION_BUDGET)
        self.assertTrue(record.success)
        self.assertEqual(record.transitions[-1].after.budget, 0)
        self.assertAlmostEqual(record.total_reward, 2.76)

    def test_partial_collect_on_last_action_is_timeout(self):
        env = LineCollectEnv()
        env.reset(EpisodeSpec(0, (1, 2, 3)))
        for _ in range(22):
            env.step(LEFT)
        env.step(RIGHT)
        result = env.step(COLLECT)
        self.assertTrue(result.terminated)
        self.assertTrue(result.collected)
        self.assertEqual(env.state.collected, 1)
        self.assertIsNone(result.observation)
        self.assertIsNone(env.state.goal)
        with self.assertRaises(RuntimeError):
            env.step(COLLECT)
        with self.assertRaises(RuntimeError):
            env.observe()

    def test_bad_inputs_fail_without_mutating_environment(self):
        env = LineCollectEnv()
        with self.assertRaises(RuntimeError):
            env.step(LEFT)
        for start, targets in [(-1, (1, 2, 3)), (True, (1, 2, 3)), (0, (0, 1, 2)),
                               (0, (1, 1, 2)), (0, (1, 2, 7)), (0, (1, 2))]:
            with self.assertRaises(ValueError):
                EpisodeSpec(start, targets)
        env.reset(EpisodeSpec(0, (1, 2, 3)))
        before = env.state
        for action in [-1, 3, True, 1.0, math.nan, None]:
            with self.assertRaises(ValueError):
                env.step(action)
            self.assertEqual(env.state, before)

    def test_actions_change_observations_and_only_four_values_reach_policy(self):
        episode = EpisodeSpec(2, (4, 1, 5))
        left, right = LineCollectEnv(), LineCollectEnv()
        left.reset(episode)
        right.reset(episode)
        left_observation = left.step(LEFT).observation
        right_observation = right.step(RIGHT).observation
        self.assertNotEqual(left_observation, right_observation)
        self.assertEqual(left_observation[1:], right_observation[1:])
        seen = []

        class SpyPolicy(CoordinatePolicy):
            def act(self, observation, rng):
                seen.append(observation)
                return super().act(observation, rng)

        run_episode(SpyPolicy(), episode, 1)
        self.assertTrue(seen)
        self.assertTrue(all(type(obs) is tuple and len(obs) == 4 for obs in seen))
        self.assertEqual(seen[0], (2 / 6, 4 / 6, 1.0, 1.0))


class DatasetAndRolloutTests(unittest.TestCase):
    def test_split_is_complete_disjoint_grouped_and_reflection_balanced(self):
        split = make_split()
        all_ids = set()
        all_goals = set()
        for name, expected_count, expected_goals in [
            ("train", 756, 126), ("validation", 252, 42), ("test", 504, 84),
        ]:
            episodes = getattr(split, name)
            ids = {episode.episode_id for episode in episodes}
            counts = Counter(episode.targets for episode in episodes)
            self.assertEqual(len(episodes), expected_count)
            self.assertEqual(len(ids), expected_count)
            self.assertEqual(len(counts), expected_goals)
            self.assertEqual(set(counts.values()), {6})
            self.assertTrue(all(reflect(goals) in counts for goals in counts))
            self.assertFalse(all_ids & ids)
            self.assertFalse(all_goals & counts.keys())
            all_ids.update(ids)
            all_goals.update(counts)
        self.assertEqual(all_ids, {episode.episode_id for episode in all_episodes()})

    def test_split_and_patterns_do_not_consume_global_random(self):
        before = random.getstate()
        split = make_split()
        self.assertEqual(split.manifest(), make_split().manifest())
        self.assertEqual(split.digest, make_split().digest)
        self.assertNotEqual(split.digest, make_split(71002).digest)
        patterns = blind_patterns()
        self.assertEqual(len(patterns), 140)
        self.assertEqual(patterns, blind_patterns())
        self.assertEqual(random.getstate(), before)

    def test_rollout_resets_policy_and_is_independent_of_previous_episodes(self):
        policy = FixedSequencePolicy((RIGHT, COLLECT, LEFT, COLLECT))
        episode = EpisodeSpec(0, (1, 0, 1))
        first = run_episode(policy, episode, derive_seed("check", 1, episode.episode_id))
        run_episode(policy, EpisodeSpec(5, (3, 2, 4)), 20)
        again = run_episode(policy, episode, derive_seed("check", 1, episode.episode_id))
        self.assertEqual(first, again)
        self.assertTrue(first.success)
        random_policy = UniformRandomPolicy()
        before = random.getstate()
        first_random = run_episode(random_policy, episode, 1234)
        run_episode(random_policy, EpisodeSpec(5, (3, 2, 4)), 50)
        self.assertEqual(first_random, run_episode(random_policy, episode, 1234))
        self.assertEqual(random.getstate(), before)
        self.assertNotEqual(derive_seed("1", 2), derive_seed(1, "2"))


if __name__ == "__main__":
    unittest.main()
