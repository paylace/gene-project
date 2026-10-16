"""실험 조건과 순서 보존 번식. 네트워크 의존성은 기존 동결 구현이다."""

from dataclasses import dataclass
import math

from evosim_interaction_v02.genetics import next_population as original_next_population
from evosim_interaction_v02.settings import Settings, digest_json

CONDITIONS = ("R0S0", "R1S0", "R0S1", "R1S1")
NETWORK = Settings("A0")


@dataclass(frozen=True)
class Treatment:
    name: str

    def __post_init__(self):
        if self.name not in CONDITIONS:
            raise ValueError("Unknown Q1-005 treatment")

    @property
    def shaping(self):
        return self.name[1] == "1"

    @property
    def completion_first(self):
        return self.name[3] == "1"

    def to_dict(self):
        return {"name": self.name, "network": NETWORK.to_dict(), "progress_unit_denominator": 60,
                "shaping": self.shaping, "completion_first": self.completion_first,
                "maximum_progress_units": 18, "lifetime_learning": False}

    @property
    def digest(self):
        return digest_json(self.to_dict())

    @classmethod
    def from_dict(cls, data):
        result = cls(data["name"])
        if result.to_dict() != data:
            raise ValueError("Treatment setting changed")
        return result


def fitness_key(metrics, treatment):
    score = metrics["mean_training_score"]
    return (metrics["success_rate"], score) if treatment.completion_first else (score,)


def ranks_for(keys):
    keys = [tuple(key) for key in keys]
    if not keys or len({len(key) for key in keys}) != 1 or any(not key for key in keys):
        raise ValueError("Nonempty equal-length fitness keys required")
    if any(not math.isfinite(value) for key in keys for value in key):
        raise ValueError("Nonfinite fitness")
    lookup = {key: rank for rank, key in enumerate(sorted(set(keys)))}
    return [lookup[key] for key in keys]


def breed(population, keys, rng, allocator):
    # 기존 토너먼트·우성 부모 선택은 순서와 동률만 사용하므로 이 치환은 동치다.
    return original_next_population(population, ranks_for(keys), rng, allocator, NETWORK)


def ranking_diagnostics(metrics, treatment):
    score = lambda row: row["mean_training_score"]
    def order(key):
        return sorted(range(len(metrics)), key=lambda slot: (*(-v for v in key(metrics[slot])), slot))
    s0 = order(lambda row: (score(row),))
    s1 = order(lambda row: (row["success_rate"], score(row)))
    base = order(lambda row: (row["mean_reward"],))
    conflicts = sum((a["success_rate"] - b["success_rate"]) * (score(a) - score(b)) < 0
                    for i, a in enumerate(metrics) for b in metrics[i + 1:])
    elites = max(1, int(len(metrics) * 0.2))
    return {"s0_order": s0, "s1_order": s1, "base_reward_order": base,
            "completion_score_conflict_pairs": conflicts, "s0_s1_order_changed": s0 != s1,
            "s0_s1_elite_sets_changed": set(s0[:elites]) != set(s1[:elites]),
            "reward_order_changed": base != s0,
            "training_candidates_with_success": sum(row["success_rate"] > 0 for row in metrics),
            "applied_order": s1 if treatment.completion_first else s0}
