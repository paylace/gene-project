# --- 전역 혁신 번호 추적기 (NEAT 스타일 교차 정렬용) ---
class InnovationTracker:
    """구조적 변이(모듈/간선)의 역사적 기원을 추적하여 경쟁적 관례(Competing Conventions) 충돌 방지"""

    def __init__(self):
        self._counter = 0
        self._registry = {}

    def get_id(self, key):
        if key not in self._registry:
            self._counter += 1
            self._registry[key] = self._counter
        return self._registry[key]


INNOVATION_TRACKER = InnovationTracker()
