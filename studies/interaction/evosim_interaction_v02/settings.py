"""조건별 불변 설정. 원래 제안 분포와 실제 허용 구조를 구분한다."""

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType

OPS = ("linear", "threshold", "relu", "tanh", "ema", "latch")
MEMORY_OPS = frozenset(("ema", "latch"))
BASELINE = MappingProxyType({
    "input_count": 4, "output_count": 3, "initial_modules": 4, "max_modules": 9,
    "elite_ratio": 0.2, "crossover_rate": 0.65, "mutation_rate": 0.50,
    "tournament_size": 3, "complexity_penalty": 0.0,
    "recurrence_mode": "seeded_positive", "allow_recurrent_edges": True,
    "allow_module_memory": True, "lifetime_learning": False,
    "td_weight": 0.0, "hebbian_lr": 0.0,
    "mutation_weights": (0.42, 0.12, 0.06, 0.18, 0.22, 0.10),
    "module_ops": OPS, "module_bias_initial": (-0.4, 0.4),
    "module_decay_initial": (0.2, 0.9), "input_weight_initial": (-1.5, 1.5),
    "chain_weight_initial": (-1.2, 1.2), "initial_self_rec_probability": 0.7,
    "initial_self_rec_weight": (0.4, 1.1), "output_primary_weight": (0.6, 1.5),
    "output_aux_weight": (-1.0, 1.0), "output_module_rec_weight": (0.4, 1.2),
    "output_self_rec_weight": (0.5, 1.2), "weight_change_probability": 0.35,
    "weight_change_sigma": 0.35, "weight_bounds": (-3.0, 3.0),
    "module_op_change_probability": 0.25, "module_bias_change_probability": 0.35,
    "module_bias_change_sigma": 0.25, "module_bias_bounds": (-2.0, 2.0),
    "module_decay_change_probability": 0.35, "module_decay_change_sigma": 0.15,
    "module_decay_bounds": (0.0, 0.95), "new_module_self_rec_probability": 0.5,
    "new_module_self_rec_weight": (0.4, 1.0), "new_edge_weight": (-1.2, 1.2),
    "remove_edge_rec_probability": 0.4, "matching_gene_second_probability": 0.5,
    "second_parent_ff_probability": 0.3, "second_parent_rec_probability": 0.35,
    "tie_rel_tol": 1e-9, "tie_abs_tol": 1e-9,
})


def digest_json(value):
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class Settings:
    condition: str

    def __post_init__(self):
        if self.condition not in ("A0", "A1", "A2"):
            raise ValueError("조건은 A0/A1/A2여야 합니다")

    @property
    def parameters(self):
        return BASELINE

    @property
    def allowed_ops(self):
        return tuple(op for op in OPS if self.condition != "A2" or op not in MEMORY_OPS)

    def allows_rec(self, source, target):
        return self.condition != "A2" and not (
            self.condition == "A1" and source == target and target.startswith("out_"))

    def to_dict(self):
        # 새 JSON 객체이므로 호출자의 수정이 설정에 영향을 주지 않는다.
        baseline = {key: list(value) if isinstance(value, tuple) else value for key, value in BASELINE.items()}
        return {"condition": self.condition, "baseline_parameters": baseline,
                "constraints": {"output_self_recurrence": self.condition == "A0",
                                "other_recurrence": self.condition != "A2",
                                "module_memory": self.condition != "A2",
                                "allowed_ops": list(self.allowed_ops)}}

    @property
    def digest(self):
        return digest_json(self.to_dict())

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or "condition" not in data:
            raise ValueError("설정 직렬화 형식 오류")
        settings = cls(data["condition"])
        if digest_json(data) != settings.digest:
            raise ValueError("지원하지 않는 설정 변경 또는 누락")
        return settings
