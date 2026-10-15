import datetime
import json
import os

from .config import CONFIG


# --- 로깅 시스템 ---
class Logger:
    def __init__(self, log_dir="simulation_logs", prefix="working_memory_sim"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{prefix}_{timestamp}.log")
        self.json_file = os.path.join(log_dir, f"{prefix}_{timestamp}.json")

        self.generation_data = []
        self._write_header()

    def _write_header(self):
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("=" * 88 + "\n")
            f.write("진화 시뮬레이션 로그 (한계점 보완 + 순환 연결 및 작업 기억 창발 연구)\n")
            f.write(f"시작 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 88 + "\n\n")

    def log_generation(
        self,
        generation,
        stage,
        avg_reward,
        avg_accuracy,
        cue_accuracy,
        memory_accuracy,
        best_reward,
        best_accuracy,
        avg_recurrent_edges,
        population_diversity,
    ):
        """세대별 로그 (단서 제시 구간 정확도 vs 마스킹 지연 구간 작업 기억 정확도 분리 기록)"""
        msg = (
            f"Generation {generation + 1:3d} | Stage {stage + 1}\n"
            f"  평균 보상: {avg_reward:6.3f} | 전체 정확도: {avg_accuracy:6.2%} "
            f"(단서 구간: {cue_accuracy:6.2%}, 기억 유지 구간: {memory_accuracy:6.2%})\n"
            f"  최고 보상: {best_reward:6.3f} | 최고 정확도: {best_accuracy:6.2%} | "
            f"평균 순환 간선: {avg_recurrent_edges:.2f}개 | 집단 다양성: {population_diversity:.3f}\n"
        )
        self._append_log(msg)

        self.generation_data.append(
            {
                "generation": generation + 1,
                "stage": stage + 1,
                "avg_reward": round(avg_reward, 4),
                "avg_accuracy": round(avg_accuracy, 4),
                "cue_accuracy": round(cue_accuracy, 4),
                "memory_accuracy": round(memory_accuracy, 4),
                "best_reward": round(best_reward, 4),
                "best_accuracy": round(best_accuracy, 4),
                "avg_recurrent_edges": round(avg_recurrent_edges, 3),
                "diversity": round(population_diversity, 4),
            }
        )

    def log_stage_advance(self, stage, generation):
        msg = f"🎓 [Gen {generation + 1}] 다음 단계로 진행! -> Stage {stage + 1} (마스킹 지연 스텝 확대)\n"
        self._append_log(msg)

    def log_final_summary(
        self, best_fitness, best_accuracy, final_stage, total_generations, comparison_data=None
    ):
        msg = (
            f"\n" + "=" * 88 + "\n"
            f"시뮬레이션 완료\n"
            f"총 세대 수: {total_generations}\n"
            f"최종 최고 적합도: {best_fitness:.3f} (최고 정확도: {best_accuracy:.2%})\n"
            f"최종 도달 단계: {final_stage + 1}/{len(CONFIG['CURRICULUM_STAGES'])}\n"
            f"종료 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"=" * 88 + "\n"
        )
        self._append_log(msg)
        self._save_json(comparison_data)

    def log_entity_structure(self, entity, label="최고 개체"):
        ops = {k: f"{m.op}(leak={m.memory_decay:.2f})" for k, m in entity.modules.items()}
        feedforward_count = sum(len(v) for v in entity.connections.values())
        recurrent_count = sum(len(v) for v in entity.recurrent_connections.values())

        msg = (
            f"\n[{label} 구조 분석]\n"
            f"  모듈 개수: {len(entity.modules)} | 순방향 간선: {feedforward_count} | 순환(메모리) 간선: {recurrent_count}\n"
            f"  모듈 연산자 및 기억 유지율: {ops}\n"
            f"  순방향 연결(t -> t): { {k: [(s, round(w, 2)) for s, w in v.items()] for k, v in entity.connections.items() if v} }\n"
            f"  순환 연결(t-1 -> t): { {k: [(s, round(w, 2)) for s, w in v.items()] for k, v in entity.recurrent_connections.items() if v} }\n"
        )
        self._append_log(msg)

    def _append_log(self, message):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def _save_json(self, comparison_data=None):
        data = {
            "config": CONFIG,
            "generations": self.generation_data,
        }
        if comparison_data is not None:
            data["ablation_comparison"] = comparison_data

        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
