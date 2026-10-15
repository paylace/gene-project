"""
1단계(v2) 공동 검증 및 재현 스크립트 (reproduce_phase1_verification.py)

실행 방법:
    python reproduce_phase1_verification.py

포함된 검증 (GPT 6.1 검토 보고서 2~6절 반영):
1. 독립 프로세스 간 PYTHONHASHSEED 결정성 검증
2. 추가 검증(enable_validation) ON/OFF 시 훈련 궤적·번식·승급 완전 불변성 검증
3. 통일된 선발 범위(집단 전체 20개체) 및 통일 지표(fitness, accuracy, memory_accuracy) 하의
   - 0세대(학습 전 vs 학습 후 동일 개체 짝 비교 및 집단 선발 비교)
   - 100세대 최종 세대 후보(Gen 100 Final Candidate) vs 1~100세대 누적 검증 후보(Historical Candidate, 선발 세대/단계 명시)
   - 세대 훈련 1위(gen_best) 한정 검증 후보 비교
4. 정밀화된 Q((0,0,0), a) 추적 로그 (초기 Q값 포함, 회상 배치 내 개별 업데이트 피크 vs 배치 종료 후 피크 분리,
   신경망 단독 동률 tie_breaks vs 순수 행동 역전 strict_flips 분리)
"""

import json
import math
import os
import random
import statistics
import subprocess
import sys

from evosim import CONFIG, Environment, EvolutionarySimulator
from evosim.simulator import candidate_selection_key


def verify_cross_process_determinism():
    print("=" * 88)
    print("[검증 1] 독립 프로세스 간 PYTHONHASHSEED 결정성 테스트 (시드 42, 30세대)")
    print("=" * 88)
    code = (
        "import json, random; "
        "from evosim import EvolutionarySimulator; "
        "random.seed(42); "
        "sim = EvolutionarySimulator(allow_recurrent=True, enable_validation=True); "
        "summary = sim.run(max_generations=30); "
        "print(json.dumps({'summary': summary, 'gens': sim.logger.generation_data}, sort_keys=True))"
    )
    root_dir = os.path.dirname(os.path.abspath(__file__))
    payloads = {}
    for hs in ("0", "42", "77777"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = hs
        res = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root_dir,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        payloads[hs] = json.loads(res.stdout.strip())
        s = payloads[hs]["summary"]
        print(
            f"  PYTHONHASHSEED={hs:<5s} -> final_trained_stage={s['final_trained_stage']}, "
            f"unlocked_stage={s['unlocked_stage']}, best_fitness={s['best_fitness']}, "
            f"val_mem_acc={s['validation_candidate']['val_memory_accuracy']}"
        )

    assert payloads["0"] == payloads["42"] == payloads["77777"], "프로세스 간 결정성 불일치!"
    print("  => 결과: PASS (서로 다른 PYTHONHASHSEED 프로세스 간 훈련 궤적 및 검증 지표 100% 일치)\n")


def verify_validation_isolation():
    print("=" * 88)
    print("[검증 2] 추가 검증(enable_validation) ON/OFF 시 훈련 과정 무간섭 테스트 (시드 42, 30세대)")
    print("=" * 88)
    random.seed(42)
    sim_off = EvolutionarySimulator(allow_recurrent=True, enable_validation=False)
    sum_off = sim_off.run(max_generations=30)

    random.seed(42)
    sim_on = EvolutionarySimulator(allow_recurrent=True, enable_validation=True)
    sum_on = sim_on.run(max_generations=30)

    assert sim_off.logger.generation_data == sim_on.logger.generation_data
    for e1, e2 in zip(sim_off.population, sim_on.population):
        assert dict(e1.connections) == dict(e2.connections)
        assert dict(e1.recurrent_connections) == dict(e2.recurrent_connections)
        assert dict(e1.value_estimates) == dict(e2.value_estimates)

    print(f"  enable_validation=False 최고 적합도: {sum_off['best_fitness']}")
    print(f"  enable_validation=True  최고 적합도: {sum_on['best_fitness']}")
    print("  => 결과: PASS (검증 삽입 전후 30세대 전체 훈련 수치 및 개체 가중치/Q값 100% 동일)\n")


def _select_best_on_val(snapshots, val_env, val_trajs, val_seed):
    """20개체 스냅샷 풀에서 통일된 candidate_selection_key(fitness, accuracy, memory_accuracy)로 최고 개체 인덱스 선발"""
    best_idx = -1
    best_key = (float("-inf"), float("-inf"), float("-inf"))
    for idx, snap in enumerate(snapshots):
        m = snap.evaluate_frozen(
            val_env, trajectories=val_trajs, rng=random.Random(val_seed)
        )
        k = candidate_selection_key(m)
        if k >= best_key:
            best_key = k
            best_idx = idx
    return best_idx


def compare_gen0_vs_evolved(seeds=(42, 43, 44, 45, 46), total_gens=100):
    print("=" * 88)
    print(
        f"[검증 3] 통일된 선발 규칙(집단 20개체 전수, (fitness, acc, mem_acc)) 및 짝 비교 하의 "
        f"0세대 vs {total_gens}세대 동결 테스트 비교 ({len(seeds)}개 시드)"
    )
    print("=" * 88)
    stage3_env = Environment(CONFIG["CURRICULUM_STAGES"][-1])
    val_trajs = stage3_env.generate_trajectories(
        CONFIG["VALIDATION_EPISODES"], seed=CONFIG["VALIDATION_SEED"], shuffle_classes=True
    )
    test_trajs = stage3_env.generate_trajectories(
        CONFIG["TEST_EPISODES"], seed=CONFIG["TEST_SEED"], shuffle_classes=True
    )

    stats = {
        "gen0_pre_selected_pre_test": [],
        "gen0_pre_selected_post_test_paired": [],
        "gen0_post_selected_post_test": [],
        "gen0_pop_mean_paired_delta": [],
        "evol_hist_pop_val_test": [],
        "evol_hist_pop_val_ep_full": [],
        "evol_hist_leader_val_test": [],
        "evol_final_gen100_val_test": [],
        "evol_final_gen100_paired_delta": [],
        "ff_hist_pop_val_test": [],
    }

    for seed in seeds:
        random.seed(seed)
        rec_sim = EvolutionarySimulator(
            allow_recurrent=True,
            enable_validation=True,
            val_trajectories=val_trajs,
            validation_scope="population",
        )
        rec_sim.run(max_generations=total_gens)

        # 1) 0세대 학습 전 선발 후보 및 동일 개체의 18ep 학습 후 짝(paired) 성능
        idx_gen0_pre = _select_best_on_val(
            rec_sim.gen0_pre_learning_snapshots, stage3_env, val_trajs, CONFIG["VALIDATION_SEED"]
        )
        g0_pre_test = rec_sim.gen0_pre_learning_snapshots[idx_gen0_pre].evaluate_frozen(
            stage3_env, trajectories=test_trajs, rng=random.Random(CONFIG["TEST_SEED"] + 1)
        )
        g0_pre_cand_after_learn_test = rec_sim.gen0_post_learning_snapshots[idx_gen0_pre].evaluate_frozen(
            stage3_env, trajectories=test_trajs, rng=random.Random(CONFIG["TEST_SEED"] + 1)
        )

        # 2) 0세대 18ep 학습 후 통일 기준으로 선발된 후보
        idx_gen0_post = _select_best_on_val(
            rec_sim.gen0_post_learning_snapshots, stage3_env, val_trajs, CONFIG["VALIDATION_SEED"]
        )
        g0_post_test = rec_sim.gen0_post_learning_snapshots[idx_gen0_post].evaluate_frozen(
            stage3_env, trajectories=test_trajs, rng=random.Random(CONFIG["TEST_SEED"] + 1)
        )

        # 3) 0세대 20개체 전원의 생애 학습 전->후 짝 차이(Paired Delta) 평균
        g0_deltas = []
        for pre_s, post_s in zip(
            rec_sim.gen0_pre_learning_snapshots, rec_sim.gen0_post_learning_snapshots
        ):
            m_pre = pre_s.evaluate_frozen(
                stage3_env, trajectories=test_trajs, rng=random.Random(CONFIG["TEST_SEED"] + 1)
            )["memory_accuracy"]
            m_post = post_s.evaluate_frozen(
                stage3_env, trajectories=test_trajs, rng=random.Random(CONFIG["TEST_SEED"] + 1)
            )["memory_accuracy"]
            g0_deltas.append(m_post - m_pre)

        # 4) 1~100세대 누적 검증 후보 (전수 범위 vs 세대 훈련 1위 범위) 및 100세대 최종 세대 후보
        hist_pop_test = rec_sim.best_val_entity.evaluate_frozen(
            stage3_env, trajectories=test_trajs, rng=random.Random(CONFIG["TEST_SEED"] + 1)
        )
        hist_leader_test = rec_sim.best_gen_leader_val_entity.evaluate_frozen(
            stage3_env, trajectories=test_trajs, rng=random.Random(CONFIG["TEST_SEED"] + 1)
        )
        final_gen_test = rec_sim.final_gen_val_entity.evaluate_frozen(
            stage3_env, trajectories=test_trajs, rng=random.Random(CONFIG["TEST_SEED"] + 1)
        )
        final_gen_pre_test = rec_sim.final_gen_val_pre_lifetime_entity.evaluate_frozen(
            stage3_env, trajectories=test_trajs, rng=random.Random(CONFIG["TEST_SEED"] + 1)
        )

        # 5) 순방향 대조군 (명시적 플래그 격리 확인)
        random.seed(seed)
        ff_sim = EvolutionarySimulator(
            allow_recurrent=False,
            allow_recurrent_edges=False,
            allow_module_memory=False,
            enable_validation=True,
            val_trajectories=val_trajs,
            validation_scope="population",
        )
        ff_sim.run(max_generations=total_gens)
        ff_test = ff_sim.best_val_entity.evaluate_frozen(
            stage3_env, trajectories=test_trajs, rng=random.Random(CONFIG["TEST_SEED"] + 1)
        )

        stats["gen0_pre_selected_pre_test"].append(g0_pre_test["memory_accuracy"])
        stats["gen0_pre_selected_post_test_paired"].append(
            g0_pre_cand_after_learn_test["memory_accuracy"]
        )
        stats["gen0_post_selected_post_test"].append(g0_post_test["memory_accuracy"])
        stats["gen0_pop_mean_paired_delta"].append(statistics.mean(g0_deltas))
        stats["evol_hist_pop_val_test"].append(hist_pop_test["memory_accuracy"])
        stats["evol_hist_pop_val_ep_full"].append(
            hist_pop_test["episode_full_memory_accuracy"]
        )
        stats["evol_hist_leader_val_test"].append(hist_leader_test["memory_accuracy"])
        stats["evol_final_gen100_val_test"].append(final_gen_test["memory_accuracy"])
        stats["evol_final_gen100_paired_delta"].append(
            final_gen_test["memory_accuracy"] - final_gen_pre_test["memory_accuracy"]
        )
        stats["ff_hist_pop_val_test"].append(ff_test["memory_accuracy"])

        print(
            f"  Seed {seed}: "
            f"Gen0[선발(전)={g0_pre_test['memory_accuracy']:.2%} -> 동일개체학습후={g0_pre_cand_after_learn_test['memory_accuracy']:.2%}, "
            f"선발(후)={g0_post_test['memory_accuracy']:.2%}] | "
            f"누적검증후보(전수, Gen {rec_sim.best_val_generation + 1}/Stage {rec_sim.best_val_trained_stage})="
            f"{hist_pop_test['memory_accuracy']:.2%}(완주 {hist_pop_test['episode_full_memory_accuracy']:.2%}) | "
            f"누적검증후보(훈련1위, Gen {rec_sim.best_gen_leader_val_generation + 1}/Stage {rec_sim.best_gen_leader_val_stage})="
            f"{hist_leader_test['memory_accuracy']:.2%} | "
            f"Gen100최종후보={final_gen_test['memory_accuracy']:.2%}(학습전후Δ={final_gen_test['memory_accuracy'] - final_gen_pre_test['memory_accuracy']:+.2%}) | "
            f"FF대조군={ff_test['memory_accuracy']:.2%}"
        )

    def _fmt(vals):
        return f"{statistics.mean(vals):.2%} ± {statistics.stdev(vals):.2%}"

    print("\n  [5개 시드 Stage 3 동결 테스트셋(120 ep) 통일 기준 요약]")
    print(f"  1) Gen 0 학습 전 선발 후보 (학습 전 성능): {_fmt(stats['gen0_pre_selected_pre_test'])}")
    print(f"  2) Gen 0 학습 전 선발 후보의 18ep 학습 후 성능 (동일 개체 짝 비교): {_fmt(stats['gen0_pre_selected_post_test_paired'])}")
    print(f"  3) Gen 0 20개체 전원의 18ep 생애 학습 전->후 평균 변화량(Δ): {_fmt(stats['gen0_pop_mean_paired_delta'])}")
    print(f"  4) Gen 0 18ep 학습 후 선발 후보 (통일 선발 기준): {_fmt(stats['gen0_post_selected_post_test'])}")
    print(f"  5) 1~100세대 누적 검증 후보 (매 세대 20개체 전수 범위): {_fmt(stats['evol_hist_pop_val_test'])} (에피소드 3스텝 완주율: {_fmt(stats['evol_hist_pop_val_ep_full'])})")
    print(f"  6) 1~100세대 누적 검증 후보 (매 세대 훈련 1위 개체 범위): {_fmt(stats['evol_hist_leader_val_test'])}")
    print(f"  7) 100세대 마지막 집단(Gen 100) 검증 선발 후보: {_fmt(stats['evol_final_gen100_val_test'])} (100세대 후보 생애 학습 전->후 Δ: {_fmt(stats['evol_final_gen100_paired_delta'])})")
    print(f"  8) 순방향(FF) 대조군 누적 검증 후보 (전수 범위): {_fmt(stats['ff_hist_pop_val_test'])}\n")


def reproduce_q_peak_and_flips_log():
    print("=" * 88)
    print(
        "[검증 4] 정밀화된 레거시 비동결 평가(120 ep, learn=True) Q((0,0,0), a) 추적 (시드 0~14)\n"
        "  - init_max_Q: 평가 시작 전 초기 Q((0,0,0), a) 최대값\n"
        "  - post_batch_max_Q: 일반 TD 업데이트 직후 및 회상 배치 종료 후 관측된 최대값\n"
        "  - all_step_max_Q: 회상 배치 내부의 개별 샘플 업데이트까지 모두 포함한 진정한 전체 최대값\n"
        "  - strict_flips vs tie_breaks: 신경망 단독 확률이 유일 최대(strict)일 때 뒤집힌 횟수 vs 동률 해소 횟수"
    )
    print("=" * 88)
    zero_key = (0.0, 0.0, 0.0)
    stage3_env = Environment(CONFIG["CURRICULUM_STAGES"][-1])

    for seed in range(15):
        random.seed(seed)
        sim = EvolutionarySimulator(allow_recurrent=True, enable_validation=False)
        sim.run(max_generations=30)
        best = sim.best_entity.copy(preserve_values=True, decay_values=True)

        init_qs = [best.value_estimates.get((zero_key, a), 0.0) for a in range(3)]
        init_max_q = max(init_qs)

        state = {
            "post_batch_max_q": init_max_q,
            "post_batch_rec": ("init", 0, -1, int( init_qs.index(init_max_q) ), round(init_max_q, 4)),
            "all_step_max_q": init_max_q,
            "all_step_rec": ("init", 0, -1, int( init_qs.index(init_max_q) ), round(init_max_q, 4)),
            "t0_hits": 0,
            "step": 0,
            "strict_flips": 0,
            "tie_breaks": 0,
        }
        orig_td = best.td_and_hebbian_update
        orig_select = best.select_action

        def spy_select(state_key, logits, epsilon=0.0, rng=None):
            act, probs = orig_select(state_key, logits, epsilon=epsilon, rng=rng)
            max_p = max(probs)
            nn_best = [i for i, p in enumerate(probs) if math.isclose(p, max_p, abs_tol=1e-9)]
            if len(nn_best) > 1:
                if act != nn_best[0]:
                    state["tie_breaks"] += 1
            else:
                if act != nn_best[0]:
                    state["strict_flips"] += 1
            return act, probs

        def spy_td(state_key, action, reward, next_state_key, done, traces=None):
            t = state["step"] % 4
            state["step"] += 1
            if state_key == zero_key and t == 0:
                state["t0_hits"] += 1
            before = best.value_estimates.get((zero_key, action), 0.0)
            err = orig_td(state_key, action, reward, next_state_key, done, traces=traces)
            if state_key == zero_key:
                q_now = best.value_estimates.get((zero_key, action), 0.0)
                rec = (
                    "td",
                    (state["step"] - 1) // 4,
                    t,
                    action,
                    round(reward, 4),
                    done,
                    round(before, 4),
                    round(q_now, 4),
                )
                if q_now > state["post_batch_max_q"]:
                    state["post_batch_max_q"] = q_now
                    state["post_batch_rec"] = rec
                if q_now > state["all_step_max_q"]:
                    state["all_step_max_q"] = q_now
                    state["all_step_rec"] = rec
            return err

        def spy_replay():
            if len(best.experience_buffer) < 12:
                return
            batch = random.sample(
                best.experience_buffer, min(16, len(best.experience_buffer))
            )
            lr = CONFIG["TD_LEARNING_RATE"]
            gamma = CONFIG["TD_DISCOUNT"]
            for s_idx, (state_key, action, reward, next_state_key, done) in enumerate(batch):
                current_q = best.value_estimates.get((state_key, action), 0.0)
                next_max_q = (
                    0.0
                    if (done or next_state_key is None)
                    else max(
                        best.value_estimates.get((next_state_key, a), 0.0)
                        for a in range(3)
                    )
                )
                new_q = current_q + lr * (reward + gamma * next_max_q - current_q)
                best.value_estimates[(state_key, action)] = new_q
                if state_key == zero_key and new_q > state["all_step_max_q"]:
                    state["all_step_max_q"] = new_q
                    state["all_step_rec"] = (
                        "replay_step",
                        state["step"] // 4,
                        s_idx,
                        action,
                        round(current_q, 4),
                        round(new_q, 4),
                    )
            for a in range(3):
                q_now = best.value_estimates.get((zero_key, a), 0.0)
                if q_now > state["post_batch_max_q"]:
                    state["post_batch_max_q"] = q_now
                    state["post_batch_rec"] = (
                        "replay_end",
                        state["step"] // 4,
                        -1,
                        a,
                        round(q_now, 4),
                    )

        best.select_action = spy_select
        best.td_and_hebbian_update = spy_td
        best.replay_experience = spy_replay
        best.evaluate(stage3_env, 120, learn=True)
        fq = [round(best.value_estimates.get((zero_key, a), 0.0), 4) for a in range(3)]
        print(
            f"  Seed {seed:2d}: init_max={init_max_q:.4f} | post_batch_max={state['post_batch_max_q']:.4f} | "
            f"all_step_max={state['all_step_max_q']:.4f} | t0_hits={state['t0_hits']} | "
            f"strict_flips={state['strict_flips']:2d} (ties={state['tie_breaks']}) | final_Q={fq}"
        )


if __name__ == "__main__":
    verify_cross_process_determinism()
    verify_validation_isolation()
    compare_gen0_vs_evolved()
    reproduce_q_peak_and_flips_log()