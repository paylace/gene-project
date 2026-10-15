import copy
import random
import unittest

from evosim import CONFIG, Entity, Environment


class ResearchIntegrityTest(unittest.TestCase):
    def test_seeded_adaptation_including_replay_is_isolated_and_repeatable(self):
        random.seed(918)
        original = Entity(allow_recurrent=True)
        env = Environment(CONFIG['CURRICULUM_STAGES'][-1])
        trajectories = env.generate_trajectories(18, seed=91002)
        a, b = original.snapshot_for_eval(), original.snapshot_for_eval()
        state = random.getstate()
        a.evaluate(env, learn=True, trajectories=trajectories, rng=random.Random(91003))
        self.assertEqual(state, random.getstate())
        b.evaluate(env, learn=True, trajectories=trajectories, rng=random.Random(91003))
        self.assertEqual(state, random.getstate())
        self.assertEqual(a.connections, b.connections)
        self.assertEqual(a.recurrent_connections, b.recurrent_connections)
        self.assertEqual(a.value_estimates, b.value_estimates)
        self.assertNotEqual(a.value_estimates, original.value_estimates)

    def test_zero_plasticity_cannot_change_weights(self):
        saved = CONFIG['HEBBIAN_LR']
        try:
            CONFIG['HEBBIAN_LR'] = 0.0
            random.seed(919)
            entity = Entity(allow_recurrent=True)
            env = Environment(CONFIG['CURRICULUM_STAGES'][-1])
            before = copy.deepcopy((entity.connections, entity.recurrent_connections))
            entity.evaluate(env, num_episodes=18, rng=random.Random(91004))
            self.assertEqual(before, (entity.connections, entity.recurrent_connections))
            self.assertTrue(entity.value_estimates)
        finally:
            CONFIG['HEBBIAN_LR'] = saved
