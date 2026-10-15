import random

from .config import CONFIG
from .environment import Environment
from .simulator import EvolutionarySimulator


def run_working_memory_research():
    """
    순환 연결 허용(Recurrent + Memory Gate) 집단과 순방향 전용(Feedforward DAG) 대조군 비교 실험.
    - 대조군(ff_sim) 생성 시 allow_recurrent_edges=False, allow_module_memory=False를 명시적으로 지정하고
      실제 활성화된 기억 플래그를 검증·기록하여 전역 설정 오버라이드 오염을 방지함.
    """
    print("=" * 88)
    print("[연구 실험] 순환 연결(Recurrent Edge) 허용 여부에 따른 작업 기억(Working Memory) 창발 비교")
    print("=" * 88)

    stage3_env = Environment(CONFIG["CURRICULUM_STAGES"][-1])
    shared_val_trajectories = stage3_env.generate_trajectories(
        CONFIG["VALIDATION_EPISODES"],
        seed=CONFIG["VALIDATION_SEED"],
        shuffle_classes=True,
    )
    shared_test_trajectories = stage3_env.generate_trajectories(
        CONFIG["TEST_EPISODES"],
        seed=CONFIG["TEST_SEED"],
        shuffle_classes=True,
    )

    # 1. 실험군: 순환 연결 및 기억 게이트 허용 (Recurrent Enabled)
    random.seed(42)
    rec_sim = EvolutionarySimulator(
        allow_recurrent=True,
        log_prefix="recurrent_memory",
        enable_validation=True,
        val_trajectories=shared_val_trajectories,
    )
    rec_summary = rec_sim.run()

    # 2. 대조군: 순환 연결 및 모듈 기억 명시적 차단 (Feedforward DAG Only)
    random.seed(42)
    ff_sim = EvolutionarySimulator(
        allow_recurrent=False,
        allow_recurrent_edges=False,
        allow_module_memory=False,
        log_prefix="feedforward_baseline",
        enable_validation=True,
        val_trajectories=shared_val_trajectories,
    )
    ff_flags = ff_sim.get_active_memory_flags()
    assert not ff_flags["allow_recurrent_edges"] and not ff_flags["allow_module_memory"], (
        f"대조군 기억 플래그 오염 감지: {ff_flags}"
    )
    ff_summary = ff_sim.run()

    # 동일 최고 난이도(Stage 3: 3스텝 연속 마스킹) 독립 테스트셋에서 동결 평가(learn=False) 수행
    rec_train_best_test = rec_sim.best_entity.evaluate_frozen(
        stage3_env,
        trajectories=shared_test_trajectories,
        rng=random.Random(CONFIG["TEST_SEED"] + 1),
    )
    ff_train_best_test = ff_sim.best_entity.evaluate_frozen(
        stage3_env,
        trajectories=shared_test_trajectories,
        rng=random.Random(CONFIG["TEST_SEED"] + 1),
    )

    rec_val_cand = rec_sim.best_val_entity or rec_sim.best_entity
    ff_val_cand = ff_sim.best_val_entity or ff_sim.best_entity

    rec_val_best_test = rec_val_cand.evaluate_frozen(
        stage3_env,
        trajectories=shared_test_trajectories,
        rng=random.Random(CONFIG["TEST_SEED"] + 1),
    )
    ff_val_best_test = ff_val_cand.evaluate_frozen(
        stage3_env,
        trajectories=shared_test_trajectories,
        rng=random.Random(CONFIG["TEST_SEED"] + 1),
    )

    comparison = {
        "recurrent_enabled": {
            **rec_summary,
            "stage3_eval_total_acc": round(rec_val_best_test["accuracy"], 4),
            "stage3_eval_cue_acc": round(rec_val_best_test["cue_accuracy"], 4),
            "stage3_eval_memory_acc": round(rec_val_best_test["memory_accuracy"], 4),
            "stage3_eval_episode_full_memory_acc": round(
                rec_val_best_test["episode_full_memory_accuracy"], 4
            ),
            "stage3_frozen_test_from_train_best": {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in rec_train_best_test.items()
            },
            "stage3_frozen_test_from_val_best": {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in rec_val_best_test.items()
            },
        },
        "feedforward_only": {
            **ff_summary,
            "stage3_eval_total_acc": round(ff_val_best_test["accuracy"], 4),
            "stage3_eval_cue_acc": round(ff_val_best_test["cue_accuracy"], 4),
            "stage3_eval_memory_acc": round(ff_val_best_test["memory_accuracy"], 4),
            "stage3_eval_episode_full_memory_acc": round(
                ff_val_best_test["episode_full_memory_accuracy"], 4
            ),
            "stage3_frozen_test_from_train_best": {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in ff_train_best_test.items()
            },
            "stage3_frozen_test_from_val_best": {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in ff_val_best_test.items()
            },
        },
    }

    rec_sim.logger._save_json(comparison_data=comparison)

    best_ops = {
        k: f"{m.op}(leak={m.memory_decay:.2f})"
        for k, m in rec_val_cand.modules.items()
    }
    rec_edges = {
        k: [(s, round(w, 2)) for s, w in v.items()]
        for k, v in rec_val_cand.recurrent_connections.items()
        if v
    }

    print("\n[Stage 3 심층 작업 기억 환경 (동결 테스트셋 120 에피소드, learn=False) 검증 결과]")
    print(
        f"1. 순환 연결 허용 (Recurrent): 훈련 도달 Stage {rec_summary['final_trained_stage']}/3 "
        f"(해금 Stage {rec_summary['unlocked_stage']}/3) | "
        f"검증 선발 후보(Gen {rec_sim.best_val_generation + 1}) 테스트 정확도: {rec_val_best_test['accuracy']:.2%} "
        f"(단서 t=0: {rec_val_best_test['cue_accuracy']:.2%}, 기억 t=1~3: {rec_val_best_test['memory_accuracy']:.2%}, "
        f"에피소드 완주율: {rec_val_best_test['episode_full_memory_accuracy']:.2%})"
    )
    print(
        f"2. 순방향 전용 (Feedforward): 훈련 도달 Stage {ff_summary['final_trained_stage']}/3 "
        f"(해금 Stage {ff_summary['unlocked_stage']}/3) | "
        f"검증 선발 후보(Gen {ff_sim.best_val_generation + 1}) 테스트 정확도: {ff_val_best_test['accuracy']:.2%} "
        f"(단서 t=0: {ff_val_best_test['cue_accuracy']:.2%}, 기억 t=1~3: {ff_val_best_test['memory_accuracy']:.2%}, "
        f"에피소드 완주율: {ff_val_best_test['episode_full_memory_accuracy']:.2%})"
    )
    print(f"\n[진화된 최고 개체 작업 기억 구조]")
    print(f"  - 모듈 연산자 구성: {best_ops}")
    print(f"  - 시간 지연 순환 간선(t-1 -> t): {rec_edges}")
    print("=" * 88)
    return comparison
