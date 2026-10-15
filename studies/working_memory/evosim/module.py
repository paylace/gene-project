import math
import random


# --- 모듈 시스템 (동적 가중치 합산 + 작업 기억 게이트 연산자) ---
class Module:
    """
    개선된 개별 계산 모듈 (한계점 2 해결 + 작업 기억 연산자 추가):
    - 입력 개수 제한(기존 3개 절삭)을 없애고, 연결 간선별 가중합(weighted_input_sum)을 그대로 수용.
    - 'ema'(지수이동평균 누수 적분기) 및 'latch'(임계값 기반 기억 고정) 연산자를 지원하여
      순환 간선과 함께 다중 스텝 작업 기억(Working Memory) 형성 가능.
    """

    OPS = ["linear", "threshold", "relu", "tanh", "ema", "latch"]

    def __init__(self, op=None, bias=None, memory_decay=None):
        self.op = op if op is not None else random.choice(self.OPS)
        self.bias = bias if bias is not None else random.uniform(-0.4, 0.4)
        # 이전 시점(t-1) 모듈 내부 상태를 얼마나 유지할지 결정하는 감쇠 계수 (0: 기억 없음 ~ 0.95: 강한 보존)
        self.memory_decay = (
            memory_decay if memory_decay is not None else random.uniform(0.2, 0.90)
        )

    def compute(self, weighted_sum, prev_self_state=0.0, allow_memory=True):
        """간선 가중합(weighted_sum)과 이전 스텝 상태(prev_self_state)를 결합해 활성화 값 계산"""
        total_drive = weighted_sum + self.bias

        if self.op == "linear":
            return self._sigmoid(total_drive)
        elif self.op == "threshold":
            return 1.0 if total_drive > 0.0 else 0.0
        elif self.op == "relu":
            return max(0.0, min(2.0, total_drive))
        elif self.op == "tanh" or not allow_memory:
            return math.tanh(max(-10.0, min(10.0, total_drive)))
        elif self.op == "ema":
            # Leaky Integrator: 현재 입력 자극과 이전 시점의 기억 상태를 보간
            curr_act = math.tanh(max(-10.0, min(10.0, total_drive)))
            return (1.0 - self.memory_decay) * curr_act + self.memory_decay * prev_self_state
        elif self.op == "latch":
            # Gated Latch: 입력 자극의 절댓값이 일정 수준 이상일 때만 새 값을 기록하고,
            # 마스킹 구간(자극 약화)에서는 이전 기억(prev_self_state)을 유지
            if abs(weighted_sum) > 0.35:
                return math.tanh(max(-10.0, min(10.0, total_drive)))
            return prev_self_state * self.memory_decay
        return 0.0

    def _sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, x))))

    def mutate(self):
        if random.random() < 0.25:
            self.op = random.choice(self.OPS)
        if random.random() < 0.35:
            self.bias = max(-2.0, min(2.0, self.bias + random.gauss(0, 0.25)))
        if random.random() < 0.35:
            self.memory_decay = max(
                0.0, min(0.95, self.memory_decay + random.gauss(0, 0.15))
            )

    def to_dict(self):
        return {
            "op": self.op,
            "bias": self.bias,
            "memory_decay": self.memory_decay,
        }

    def copy(self):
        return Module(self.op, self.bias, self.memory_decay)
