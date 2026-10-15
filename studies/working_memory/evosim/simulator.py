import math
import random

from .config import CONFIG
from .entity import Entity
from .environment import Environment
from .logger import Logger


def _module_sort_key(mod_id):
    """모듈 ID('m0', 'm10' 등)의 프로세스 독립적·결정론적 정렬 키"""
    if isinstance(mod_id, str) and mod_id.startswith("m") and mod_id[1:].isdigit():
        return (0, int(mod_id[1:]))
    return (1, str(mod_id))


def candidate_selection_key(metrics):
    """모든 후보 선발(0세대, 진화 중, 최종 세대)에 공통 적용하는 통일 선발 기준"""
    return (
        metrics["fitness"],
        metrics["accuracy"],
        metrics["memory_accuracy"],
    )


# --- 진화 시뮬레이터 ---
class EvolutionarySimulator:
    def __init__(
        self,
        allow_recurrent=True,
        log_prefix="working_memory_sim",
        enable_validation=True,
        val_trajectories=None,
        val_seed=None,
        allow_recurrent_edges=None,
        allow_module_memory=None,
        validation_scope="population",
    ):
        """
        Args:
            allow_recurrent: 기존 총괄 순환/기억 허용 플래그
            allow_recurrent_edges: 그래프 순환 간선(t-1 -> t) 명시적 오버라이드 (None이면 CONFIG 또는 allow_recurrent 따름)
            allow_module_memory: 모듈 내부 기억(ema/latch) 명시적 오버라이드 (None이면 CONFIG 또는 allow_recurrent 따름)
            validation_scope:
                - "population"(기본, 통일 기준): 매 세대 20개체 전원의 스냅샷을 검증셋에서 평가해 후보 선발
                  (0세대 전수 선발과 후보 범위·선발 지표를 완전히 일치시킴)
                - "gen_best": 각 세대 훈련 최고 개체 1개만 검증셋에서 평가
        """
        self.allow_recurrent = allow_recurrent
        cfg_rec_edges = CONFIG.get("ALLOW_RECURRENT_EDGES")
        cfg_mod_mem = CONFIG.get("ALLOW_MODULE_MEMORY")

        self.allow_recurrent_edges = (
            (self.allow_recurrent if cfg_rec_edges is None else cfg_rec_edges)
            if allow_recurrent_edges is None
            else allow_recurrent_edges
        )
        self.allow_module_memory = (
            (self.allow_recurrent if cfg_mod_mem is None else cfg_mod_mem)
            if allow_module_memory is None
            else allow_module_memory
        )
        self.validation_scope = validation_scope

        self.population = [
            Entity(
                allow_recurrent=allow_recurrent,
                allow_recurrent_edges=self.allow_recurrent_edges,
                allow_module_memory=self.allow_module_memory,
            )
            for _ in range(CONFIG["POPULATION_SIZE"])
        ]
        self.current_stage = 0
        self.last_trained_stage = 0
        self.unlocked_stage = 0
        self.generation = 0

        # 1) 현재 실제 훈련 단계 기준 최고 개체 (무손실 스냅샷)
        self.best_entity = None
        self.best_fitness = float("-inf")
        self.best_accuracy = 0.0
        self.best_memory_accuracy = 0.0

        # 2) 실제 훈련된 각 단계(Stage)별 최고 개체 기록
        self.best_entity_per_stage = {}

        # 3) 공통 Stage 3 동결 검증셋(Validation Set) 기준 보고 후보 개체 기록
        self.enable_validation = enable_validation
        self.val_seed = CONFIG["VALIDATION_SEED"] if val_seed is None else val_seed
        self.val_env = Environment(CONFIG["CURRICULUM_STAGES"][-1])
        if val_trajectories is not None:
            self.val_trajectories = val_trajectories
        elif enable_validation:
            # 전역 random 상태를 건드리지 않고 전용 Random 인스턴스로 검증 궤적 생성
            self.val_trajectories = self.val_env.generate_trajectories(
                CONFIG["VALIDATION_EPISODES"],
                seed=self.val_seed,
                shuffle_classes=True,
            )
        else:
            self.val_trajectories = None

        # 3-A) 전체 세대 누적 최고 검증 후보 (Historical Best Validation Candidate)
        self.best_val_entity = None
        self.best_val_pre_lifetime_entity = None
        self.best_val_metrics = None
        self.best_val_key = (float("-inf"), float("-inf"), float("-inf"))
        self.best_val_generation = -1
        self.best_val_trained_stage = 1
        self.best_val_individual_idx = -1

        # 3-B) 각 세대의 훈련 1위(gen_best)만 대상으로 한 누적 최고 검증 후보 (비교 참조용)
        self.best_gen_leader_val_entity = None
        self.best_gen_leader_val_metrics = None
        self.best_gen_leader_val_key = (float("-inf"), float("-inf"), float("-inf"))
        self.best_gen_leader_val_generation = -1
        self.best_gen_leader_val_stage = 1

        # 3-C) 최종 세대(Final Generation) 집단 내 최고 검증 후보 (과거 시점 후보와 구분)
        self.final_gen_val_entity = None
        self.final_gen_val_pre_lifetime_entity = None
        self.final_gen_val_metrics = None
        self.final_gen_pre_learning_snapshots = None
        self.final_gen_post_learning_snapshots = None

        # 0세대 초기 집단 스냅샷 (생애 학습 전 / 1세대 생애 학습 직후 — 동일 개체 인덱스 1:1 대응)
        self.gen0_pre_learning_snapshots = [
            e.snapshot_for_eval() for e in self.population
        ]
        self.gen0_post_learning_snapshots = None

        self.logger = Logger(prefix=log_prefix)

    def get_active_memory_flags(self):
        """현재 시뮬레이터와 개체들에 실제 적용된 기억 관련 플래그 명세 반환"""
        return {
            "allow_recurrent": self.allow_recurrent,
            "allow_recurrent_edges": self.allow_recurrent_edges,
            "allow_module_memory": self.allow_module_memory,
            "initial_recurrence_mode": CONFIG.get("INITIAL_RECURRENCE_MODE", "seeded_positive"),
            "q_state_includes_memory": CONFIG.get("Q_STATE_INCLUDES_MEMORY", False),
            "td_weight": CONFIG.get("TD_WEIGHT", 0.3),
            "hebbian_lr": CONFIG.get("HEBBIAN_LR", 0.015),
        }

    def run(self, max_generations=None):
        total_gens = max_generations or CONFIG["MAX_GENERATIONS"]

        for self.generation in range(total_gens):
            trained_stage = self.current_stage
            self.last_trained_stage = trained_stage
            stage_config = CONFIG["CURRICULUM_STAGES"][trained_stage]
            env = Environment(stage_config)

            # 생애 학습(18 에피소드) 전 각 개체의 무손실 스냅샷 보존 (짝 비교용)
            gen_pre_snapshots = [e.snapshot_for_eval() for e in self.population]

            results = []
            for idx, entity in enumerate(self.population):
                fitness, acc, cue_acc, mem_acc = entity.evaluate(
                    env, CONFIG["EPISODES_PER_EVAL"], learn=True
                )
                results.append((fitness, acc, cue_acc, mem_acc, entity, idx))

            gen_post_snapshots = [e.snapshot_for_eval() for e in self.population]
            if self.generation == 0:
                self.gen0_post_learning_snapshots = gen_post_snapshots
            if self.generation == total_gens - 1:
                self.final_gen_pre_learning_snapshots = gen_pre_snapshots
                self.final_gen_post_learning_snapshots = gen_post_snapshots

            results.sort(key=lambda x: (x[0], x[1], x[3]), reverse=True)

            avg_reward = sum(r[0] for r in results) / len(results)
            avg_accuracy = sum(r[1] for r in results) / len(results)
            avg_cue_acc = sum(r[2] for r in results) / len(results)
            avg_mem_acc = sum(r[3] for r in results) / len(results)
            avg_rec_edges = sum(
                r[4].count_recurrent_edges() for r in results
            ) / len(results)

            best_fit, best_acc, best_cue_acc, best_mem_acc, gen_best, gen_best_idx = results[0]
            diversity = self._calculate_diversity()

            self.logger.log_generation(
                self.generation,
                trained_stage,
                avg_reward,
                avg_accuracy,
                avg_cue_acc,
                avg_mem_acc,
                best_fit,
                best_acc,
                avg_rec_edges,
                diversity,
            )

            # A. 현재 훈련 단계 최고 개체 갱신 (무손실 스냅샷으로 Q값 감쇠 방지)
            if best_fit >= self.best_fitness:
                self.best_fitness = best_fit
                self.best_accuracy = best_acc
                self.best_memory_accuracy = best_mem_acc
                self.best_entity = gen_post_snapshots[gen_best_idx].snapshot_for_eval()

            # B. 단계별(Stage별) 최고 훈련 개체 기록
            stage_key_tuple = (best_fit, best_acc, best_mem_acc)
            prev_stage_rec = self.best_entity_per_stage.get(trained_stage)
            if prev_stage_rec is None or stage_key_tuple >= prev_stage_rec["key"]:
                self.best_entity_per_stage[trained_stage] = {
                    "key": stage_key_tuple,
                    "generation": self.generation,
                    "stage": trained_stage + 1,
                    "entity": gen_post_snapshots[gen_best_idx].snapshot_for_eval(),
                    "pre_lifetime_entity": gen_pre_snapshots[gen_best_idx].snapshot_for_eval(),
                    "training_metrics": {
                        "fitness": best_fit,
                        "accuracy": best_acc,
                        "cue_accuracy": best_cue_acc,
                        "memory_accuracy": best_mem_acc,
                    },
                }

            # C. 공통 Stage 3 동결 검증셋을 이용한 보고 후보 개체 선발
            #    (전역 random 상태를 완벽히 보존·복원하고 독립 난수원을 사용하여 훈련 궤적에 영향 0 보장)
            if self.enable_validation and self.val_trajectories is not None:
                saved_rng_state = random.getstate()
                try:
                    # 1) 세대 훈련 1위(gen_best)에 대한 검증 평가
                    leader_snap = gen_post_snapshots[gen_best_idx].snapshot_for_eval()
                    leader_val_metrics = leader_snap.evaluate_frozen(
                        self.val_env,
                        trajectories=self.val_trajectories,
                        rng=random.Random(self.val_seed),
                    )
                    leader_snap.validation_metrics = leader_val_metrics
                    leader_key = candidate_selection_key(leader_val_metrics)
                    if leader_key >= self.best_gen_leader_val_key:
                        self.best_gen_leader_val_key = leader_key
                        self.best_gen_leader_val_entity = leader_snap
                        self.best_gen_leader_val_metrics = leader_val_metrics
                        self.best_gen_leader_val_generation = self.generation
                        self.best_gen_leader_val_stage = trained_stage + 1

                    # 2) 설정된 validation_scope("population" 또는 "gen_best")에 따른 후보 선발
                    if self.validation_scope == "population":
                        candidate_indices = range(len(gen_post_snapshots))
                    else:
                        candidate_indices = [gen_best_idx]

                    gen_top_val_key = (float("-inf"), float("-inf"), float("-inf"))
                    gen_top_snap = None
                    gen_top_pre_snap = None
                    gen_top_metrics = None
                    gen_top_idx = -1

                    for c_idx in candidate_indices:
                        if c_idx == gen_best_idx:
                            c_snap = leader_snap
                            c_metrics = leader_val_metrics
                        else:
                            c_snap = gen_post_snapshots[c_idx].snapshot_for_eval()
                            c_metrics = c_snap.evaluate_frozen(
                                self.val_env,
                                trajectories=self.val_trajectories,
                                rng=random.Random(self.val_seed),
                            )
                            c_snap.validation_metrics = c_metrics

                        c_key = candidate_selection_key(c_metrics)
                        if c_key >= gen_top_val_key:
                            gen_top_val_key = c_key
                            gen_top_snap = c_snap
                            gen_top_pre_snap = gen_pre_snapshots[c_idx].snapshot_for_eval()
                            gen_top_metrics = c_metrics
                            gen_top_idx = c_idx

                    if gen_top_val_key >= self.best_val_key:
                        self.best_val_key = gen_top_val_key
                        self.best_val_entity = gen_top_snap
                        self.best_val_pre_lifetime_entity = gen_top_pre_snap
                        self.best_val_metrics = gen_top_metrics
                        self.best_val_generation = self.generation
                        self.best_val_trained_stage = trained_stage + 1
                        self.best_val_individual_idx = gen_top_idx

                    if self.generation == total_gens - 1:
                        self.final_gen_val_entity = gen_top_snap
                        self.final_gen_val_pre_lifetime_entity = gen_top_pre_snap
                        self.final_gen_val_metrics = gen_top_metrics
                finally:
                    random.setstate(saved_rng_state)

            if (self.generation + 1) % 40 == 0 and self.best_entity is not None:
                self.logger.log_entity_structure(
                    self.best_entity, f"Generation {self.generation + 1} 최고 개체"
                )

            # 단계 승급 판정: 전체 정확도와 기억 유지 구간(masked step) 정확도가 모두 기준을 충족할 때 승급
            stage_target = stage_config.get(
                "threshold", CONFIG["STAGE_SUCCESS_THRESHOLD"]
            )
            if avg_accuracy >= stage_target and avg_mem_acc >= (stage_target - 0.05):
                if self.current_stage < len(CONFIG["CURRICULUM_STAGES"]) - 1:
                    next_stage = self.current_stage + 1
                    self.unlocked_stage = max(self.unlocked_stage, next_stage)
                    self.logger.log_stage_advance(next_stage, self.generation)
                    if self.generation < total_gens - 1:
                        self.current_stage = next_stage
                        self.best_fitness = float("-inf")

            # _evolve에 기존 5-튜플 형식 전달
            results_for_evolve = [(r[0], r[1], r[2], r[3], r[4]) for r in results]
            self.population = self._evolve(results_for_evolve)

        self.logger.log_final_summary(
            self.best_fitness,
            self.best_accuracy,
            self.last_trained_stage,
            total_gens,
        )
        if self.best_entity is not None:
            self.logger.log_entity_structure(self.best_entity, "최종 최고 개체")

        summary = {
            "final_stage": self.last_trained_stage + 1,          # 하위 호환: 실제 훈련된 최종 단계
            "final_trained_stage": self.last_trained_stage + 1,  # 실제 훈련이 수행된 최종 단계
            "unlocked_stage": self.unlocked_stage + 1,           # 승급 조건을 달성해 해금한 최고 단계
            "best_fitness": round(self.best_fitness, 4),
            "best_accuracy": round(self.best_accuracy, 4),
            "best_memory_accuracy": round(self.best_memory_accuracy, 4),
            "avg_recurrent_edges": round(
                sum(e.count_recurrent_edges() for e in self.population)
                / len(self.population),
                2,
            ),
            "active_memory_flags": self.get_active_memory_flags(),
        }
        if self.best_val_metrics is not None:
            summary["validation_candidate"] = {
                "validation_scope": self.validation_scope,
                "selected_generation": self.best_val_generation + 1,
                "trained_stage_at_selection": self.best_val_trained_stage,
                "val_fitness": round(self.best_val_metrics["fitness"], 4),
                "val_accuracy": round(self.best_val_metrics["accuracy"], 4),
                "val_cue_accuracy": round(self.best_val_metrics["cue_accuracy"], 4),
                "val_memory_accuracy": round(self.best_val_metrics["memory_accuracy"], 4),
            }
        return summary

    def _calculate_diversity(self):
        if not self.population:
            return 0.0

        signatures = []
        for e in self.population:
            mod_count = len(e.modules)
            ff_count = sum(len(v) for v in e.connections.values())
            rec_count = e.count_recurrent_edges()
            mean_decay = sum(m.memory_decay for m in e.modules.values()) / max(
                1, mod_count
            )
            signatures.append((mod_count, ff_count, rec_count, mean_decay))

        diversity_score = 0.0
        for dim in range(4):
            vals = [s[dim] for s in signatures]
            avg = sum(vals) / len(vals)
            var = sum((x - avg) ** 2 for x in vals) / len(vals)
            diversity_score += math.sqrt(var)
        return diversity_score

    def _evolve(self, results):
        new_population = []

        elite_count = max(1, int(CONFIG["POPULATION_SIZE"] * CONFIG["ELITE_RATIO"]))
        elites = [
            r[4].copy(preserve_values=True, decay_values=True)
            for r in results[:elite_count]
        ]
        new_population.extend(elites)

        while len(new_population) < CONFIG["POPULATION_SIZE"]:
            if random.random() < CONFIG["CROSSOVER_RATE"]:
                parent1 = self._tournament_select(results)
                parent2 = self._tournament_select(results)
                child = self._crossover(parent1, parent2)
            else:
                parent = self._tournament_select(results)
                child = parent.copy(preserve_values=True, decay_values=True)

            if random.random() < CONFIG["MUTATION_RATE"]:
                child.mutate()

            new_population.append(child)

        return new_population[: CONFIG["POPULATION_SIZE"]]

    def _tournament_select(self, results):
        tournament = random.sample(
            results, min(CONFIG["TOURNAMENT_SIZE"], len(results))
        )
        winner = max(tournament, key=lambda x: (x[0], x[3]))
        return winner[4]

    def _crossover(self, parent1, parent2):
        """
        교차 연산:
        - 프로세스 해시 시드(PYTHONHASHSEED)에 의존하지 않도록 공통 모듈 집합(shared_mods)을
          결정론적 순서(_module_sort_key)로 정렬하여 순회함.
        - 순환 간선 교차 여부는 self.allow_recurrent가 아니라 자식 개체의 실제 유효 그래프 순환 플래그
          (child.allow_recurrent_edges)로 제어하여 설정 분리 시 누락을 방지함.
        """
        child = parent1.copy(preserve_values=True, decay_values=True)

        # 1. 공통 모듈 교차 (정렬 순회로 프로세스 간 완전 결정성 보장)
        shared_mods = sorted(
            set(parent1.modules.keys()) & set(parent2.modules.keys()),
            key=_module_sort_key,
        )
        for mod_id in shared_mods:
            if random.random() < 0.5:
                child.modules[mod_id] = parent2.modules[mod_id].copy()

        # 2. 순방향 간선 정렬 교차
        for target, src_map in parent2.connections.items():
            if target in child.modules or target.startswith("out_"):
                for src, weight in src_map.items():
                    if src.startswith("in_") or src in child.modules:
                        if src in child.connections[target]:
                            if random.random() < 0.5:
                                child.connections[target][src] = weight
                        elif random.random() < 0.3 and not child._creates_cycle(
                            src, target
                        ):
                            child.connections[target][src] = weight

        # 3. 순환(Recurrent) 간선 정렬 교차 (child.allow_recurrent_edges 기준 제어)
        use_rec_edges = getattr(
            child,
            "allow_recurrent_edges",
            getattr(self, "allow_recurrent_edges", self.allow_recurrent),
        )
        if use_rec_edges:
            for target, src_map in parent2.recurrent_connections.items():
                if target in child.modules or target.startswith("out_"):
                    for src, weight in src_map.items():
                        if src in child.modules or src.startswith("out_"):
                            if src in child.recurrent_connections[target]:
                                if random.random() < 0.5:
                                    child.recurrent_connections[target][src] = weight
                            elif random.random() < 0.35:
                                child.recurrent_connections[target][src] = weight

        return child
