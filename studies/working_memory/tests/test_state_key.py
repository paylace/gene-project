"""버그 ② 및 1단계(측정 체계 확립 + 설정 분리 회귀 테스트) 공동 합의 인수 테스트"""

import os
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evosim import CONFIG, Entity, Environment, EvolutionarySimulator  # noqa: E402


class StateKeyTest(unittest.TestCase):
    def setUp(self):
        self._saved_flag = CONFIG["Q_STATE_INCLUDES_MEMORY"]
        random.seed(0)
        self.rec = Entity(allow_recurrent=True)
        self.ff = Entity(allow_recurrent=False)

    def tearDown(self):
        CONFIG["Q_STATE_INCLUDES_MEMORY"] = self._saved_flag

    def _prime(self, entity, cue):
        """단서를 한 번 보여줘서 내부 기억(prev_outputs)을 채운 뒤 마스킹 관측을 반환"""
        entity.reset_memory()
        entity.forward(cue)
        return [0.0, 0.0, 0.0]

    def test_default_key_is_observation_only_and_identical_across_groups(self):
        CONFIG["Q_STATE_INCLUDES_MEMORY"] = False
        masked = self._prime(self.rec, [0.9, 0.1, 0.5])
        self._prime(self.ff, [0.9, 0.1, 0.5])

        rec_key = self.rec._get_state_key(masked)
        ff_key = self.ff._get_state_key(masked)
        self.assertEqual(len(rec_key), 3)
        self.assertEqual(rec_key, ff_key)

    def test_default_key_does_not_leak_previous_cue(self):
        """마스킹 스텝의 키는 직전에 어떤 단서를 봤는지와 무관해야 함 (Q-테이블 경유 기억 차단)"""
        CONFIG["Q_STATE_INCLUDES_MEMORY"] = False
        keys = set()
        for cue in ([0.9, 0.1, 0.5], [0.3, 0.9, 0.1], [0.5, 0.5, 0.9]):
            masked = self._prime(self.rec, cue)
            keys.add(self.rec._get_state_key(masked))
        self.assertEqual(len(keys), 1)

    def test_legacy_flag_restores_memory_context(self):
        CONFIG["Q_STATE_INCLUDES_MEMORY"] = True
        masked = self._prime(self.rec, [0.9, 0.1, 0.5])
        self._prime(self.ff, [0.9, 0.1, 0.5])
        self.assertEqual(len(self.rec._get_state_key(masked)), 3 + 2 + 1)
        self.assertEqual(len(self.ff._get_state_key(masked)), 3)


class Phase1EvaluationTest(unittest.TestCase):
    def setUp(self):
        self._saved_cfg = dict(CONFIG)

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self._saved_cfg)

    def test_cross_process_hashseed_determinism(self):
        """서로 다른 PYTHONHASHSEED를 가진 독립 프로세스에서 동일한 수치 궤적이 나와야 함"""
        code = (
            "import json, random; "
            "from evosim import EvolutionarySimulator; "
            "random.seed(42); "
            "sim = EvolutionarySimulator(allow_recurrent=True, enable_validation=True); "
            "summary = sim.run(max_generations=15); "
            "print(json.dumps({'summary': summary, 'gens': sim.logger.generation_data}, sort_keys=True))"
        )
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        outputs = []
        for hash_seed in ("0", "42", "98765"):
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = hash_seed
            res = subprocess.run(
                [sys.executable, "-c", code],
                cwd=root_dir,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            outputs.append(res.stdout.strip())

        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])

    def test_validation_does_not_alter_training_trajectory(self):
        """추가 검증(enable_validation=True/False) 여부가 훈련 궤적, 번식 결과, 단계 승급을 바꾸지 않아야 함"""
        random.seed(42)
        sim_off = EvolutionarySimulator(allow_recurrent=True, enable_validation=False)
        sum_off = sim_off.run(max_generations=25)
        rng_after_off = random.getstate()

        random.seed(42)
        sim_on = EvolutionarySimulator(allow_recurrent=True, enable_validation=True)
        sum_on = sim_on.run(max_generations=25)
        rng_after_on = random.getstate()

        self.assertEqual(sim_off.logger.generation_data, sim_on.logger.generation_data)
        self.assertEqual(sum_off["final_trained_stage"], sum_on["final_trained_stage"])
        self.assertEqual(sum_off["unlocked_stage"], sum_on["unlocked_stage"])
        self.assertEqual(sum_off["best_fitness"], sum_on["best_fitness"])
        self.assertEqual(rng_after_off, rng_after_on)

        for e_off, e_on in zip(sim_off.population, sim_on.population):
            self.assertEqual(dict(e_off.connections), dict(e_on.connections))
            self.assertEqual(dict(e_off.recurrent_connections), dict(e_on.recurrent_connections))
            self.assertEqual(dict(e_off.value_estimates), dict(e_on.value_estimates))

    def test_lossless_snapshot_vs_reproduction_decay(self):
        """평가 스냅샷은 Q값을 100% 보존하고, 번식 복제는 LAMARCKIAN_DECAY(0.5)를 적용해야 함"""
        random.seed(0)
        entity = Entity(allow_recurrent=True)
        entity.value_estimates[((0.0, 0.0, 0.0), 1)] = 2.0

        snap = entity.snapshot_for_eval()
        child = entity.copy(preserve_values=True, decay_values=True)

        self.assertAlmostEqual(snap.value_estimates[((0.0, 0.0, 0.0), 1)], 2.0)
        self.assertAlmostEqual(
            child.value_estimates[((0.0, 0.0, 0.0), 1)],
            2.0 * CONFIG["LAMARCKIAN_DECAY"],
        )

    def test_frozen_evaluation_immutability_and_rng_isolation(self):
        """동결 평가(learn=False)는 가중치, Q-테이블(키 집합 포함), 경험 버퍼, 훈련 지표, 전역 RNG를 변경하지 않아야 함"""
        random.seed(7)
        env = Environment(CONFIG["CURRICULUM_STAGES"][-1])
        entity = Entity(allow_recurrent=True)
        train_fit, train_acc, train_cue, train_mem = entity.evaluate(
            env, num_episodes=6, learn=True
        )

        mods_before = {k: m.to_dict() for k, m in entity.modules.items()}
        conn_before = {k: dict(v) for k, v in entity.connections.items()}
        rec_before = {k: dict(v) for k, v in entity.recurrent_connections.items()}
        q_before = dict(entity.value_estimates)
        buf_before = list(entity.experience_buffer)

        test_trajs = env.generate_trajectories(12, seed=999, shuffle_classes=True)
        global_rng_before = random.getstate()

        entity.evaluate(
            env,
            learn=False,
            trajectories=test_trajs,
            rng=random.Random(1234),
            update_metrics=False,
        )

        global_rng_after = random.getstate()
        self.assertEqual(global_rng_before, global_rng_after)
        self.assertEqual({k: m.to_dict() for k, m in entity.modules.items()}, mods_before)
        self.assertEqual({k: dict(v) for k, v in entity.connections.items()}, conn_before)
        self.assertEqual({k: dict(v) for k, v in entity.recurrent_connections.items()}, rec_before)
        self.assertEqual(dict(entity.value_estimates), q_before)
        self.assertEqual(list(entity.experience_buffer), buf_before)
        self.assertAlmostEqual(entity.fitness, train_fit)
        self.assertAlmostEqual(entity.accuracy, train_acc)

    def test_last_generation_stage_advance_preserves_trained_stage_and_fitness(self):
        """마지막 세대에서 승급 조건을 달성해도 final_trained_stage와 best_fitness(-inf 방지)가 보존되어야 함"""
        CONFIG["CURRICULUM_STAGES"] = [
            {"noise": 0.00, "mask_steps": 1, "threshold": 0.0},
            {"noise": 0.08, "mask_steps": 2, "threshold": 0.99},
        ]
        random.seed(42)
        sim = EvolutionarySimulator(allow_recurrent=True, enable_validation=False)
        summary = sim.run(max_generations=1)

        self.assertEqual(summary["final_trained_stage"], 1)
        self.assertEqual(summary["unlocked_stage"], 2)
        self.assertTrue(summary["best_fitness"] > float("-inf"))

    def test_decoupled_module_memory_updates_prev_outputs(self):
        """그래프 순환이 꺼지고 모듈 내부 기억만 켜졌을 때도 prev_outputs가 갱신되어 ema가 작동해야 함"""
        CONFIG["ALLOW_RECURRENT_EDGES"] = False
        CONFIG["ALLOW_MODULE_MEMORY"] = True
        random.seed(0)
        entity = Entity(allow_recurrent=False)
        for m in entity.modules.values():
            m.op = "ema"
            m.memory_decay = 0.8

        entity.reset_memory()
        entity.forward([1.0, 0.0, 0.0])
        self.assertTrue(any(abs(entity.prev_outputs.get(k, 0.0)) > 1e-6 for k in entity.modules))

    def test_crossover_uses_child_allow_recurrent_edges_and_control_isolation(self):
        """
        GPT 6.1 검토 보고서 2절 회귀 테스트:
        - allow_recurrent=False이더라도 CONFIG['ALLOW_RECURRENT_EDGES']=True이면
          자식의 allow_recurrent_edges=True에 따라 _crossover가 부모2의 순환 간선을 정상 상속해야 함.
        - 반대로 명시적 대조군(allow_recurrent_edges=False)은 전역 설정이 True여도 순환 간선을 차단해야 함.
        """
        CONFIG["ALLOW_RECURRENT_EDGES"] = True
        CONFIG["ALLOW_MODULE_MEMORY"] = False

        sim = EvolutionarySimulator(allow_recurrent=False, enable_validation=False)
        parent1 = sim.population[0]
        parent2 = sim.population[1]
        self.assertTrue(parent1.allow_recurrent_edges)
        self.assertFalse(parent1.allow_module_memory)

        parent1.recurrent_connections["out_0"].pop("out_1", None)
        parent2.recurrent_connections["out_0"]["out_1"] = 0.7

        with patch("evosim.simulator.random.random", return_value=0.0):
            child = sim._crossover(parent1, parent2)

        self.assertTrue(child.allow_recurrent_edges)
        self.assertIn("out_1", child.recurrent_connections["out_0"])
        self.assertAlmostEqual(child.recurrent_connections["out_0"]["out_1"], 0.7)

        # 명시적 대조군 격리 검증: 전역 CONFIG가 True여도 명시적 False 인자가 우선해야 함
        ctrl_sim = EvolutionarySimulator(
            allow_recurrent=False,
            allow_recurrent_edges=False,
            allow_module_memory=False,
            enable_validation=False,
        )
        flags = ctrl_sim.get_active_memory_flags()
        self.assertFalse(flags["allow_recurrent_edges"])
        self.assertFalse(flags["allow_module_memory"])
        self.assertEqual(ctrl_sim.population[0].count_recurrent_edges(), 0)


if __name__ == "__main__":
    unittest.main()
