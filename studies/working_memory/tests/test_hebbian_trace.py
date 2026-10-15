"""버그 ①: Hebbian 업데이트가 각 전이 시점의 활성 흔적(trace)을 사용하는지 검증"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evosim import CONFIG, Entity, Environment  # noqa: E402


class HebbianTraceTest(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        self.entity = Entity(allow_recurrent=True)
        # 단서 1스텝 + 마스킹 3스텝 → 스텝마다 trace가 확연히 다름
        self.env = Environment(CONFIG["CURRICULUM_STAGES"][-1])

    def test_each_transition_uses_its_own_step_trace(self):
        entity = self.entity
        produced = []   # forward()가 스텝마다 만든 trace
        consumed = []   # td_and_hebbian_update()가 실제로 사용한 trace

        orig_forward = entity.forward
        orig_update = entity.td_and_hebbian_update

        def spy_forward(stimuli):
            out = orig_forward(stimuli)
            produced.append(entity.last_traces)
            return out

        def spy_update(*args, traces=None, **kwargs):
            consumed.append(traces if traces is not None else entity.last_traces)
            return orig_update(*args, traces=traces, **kwargs)

        entity.forward = spy_forward
        entity.td_and_hebbian_update = spy_update

        entity.evaluate(self.env, num_episodes=3)

        steps = CONFIG["STEPS_PER_EPISODE"]
        self.assertEqual(len(produced), 3 * steps)
        self.assertEqual(len(consumed), 3 * steps)
        for t, (p, c) in enumerate(zip(produced, consumed)):
            self.assertIs(c, p, f"transition {t} used the wrong step's trace")

        # 같은 에피소드 안에서 단서 스텝(t=0)과 마스킹 스텝의 trace는 서로 달라야 함
        self.assertIsNot(consumed[0], consumed[steps - 1])

    def test_traces_are_not_mutated_by_later_steps(self):
        """forward()가 이전 스텝의 trace dict를 재사용/변경하지 않아야 저장된 스냅샷이 유효함"""
        entity = self.entity
        entity.reset_memory()
        entity.forward([0.9, 0.1, 0.5])
        first = entity.last_traces
        snapshot = {k: (dict(v[0]), v[1]) for k, v in first.items()}
        entity.forward([0.0, 0.0, 0.0])
        self.assertIsNot(entity.last_traces, first)
        self.assertEqual({k: (dict(v[0]), v[1]) for k, v in first.items()}, snapshot)


if __name__ == "__main__":
    unittest.main()
