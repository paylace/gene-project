"""환경 객체를 받지 않고 네 관측값만 사용하는 기준 정책."""

import random
from itertools import product
from typing import Protocol

from .environment import COLLECT, LEFT, RIGHT, Observation, valid_action


class Policy(Protocol):
    def reset(self) -> None: ...

    def act(self, observation: Observation, rng: random.Random) -> int: ...


class CoordinatePolicy:
    def reset(self):
        pass

    def act(self, observation: Observation, rng: random.Random) -> int:
        position, goal, _, _ = observation
        return RIGHT if position < goal else LEFT if position > goal else COLLECT


class UniformRandomPolicy:
    def reset(self):
        pass

    def act(self, observation: Observation, rng: random.Random) -> int:
        return rng.randrange(3)


class FixedSequencePolicy:
    """종료 전까지 관측·보상과 무관하게 같은 순서를 반복한다."""

    def __init__(self, pattern: tuple[int, ...]):
        if not pattern or any(not valid_action(action) for action in pattern):
            raise ValueError("유효한 행동의 비어 있지 않은 순서가 필요합니다.")
        self.pattern = tuple(pattern)
        self.reset()

    def reset(self):
        self._index = 0

    def act(self, observation: Observation, rng: random.Random) -> int:
        action = self.pattern[self._index % len(self.pattern)]
        self._index += 1
        return action


def blind_patterns(seed: int = 71002) -> tuple[tuple[int, ...], ...]:
    patterns = [pattern for size in range(1, 5) for pattern in product(range(3), repeat=size)]
    rng = random.Random(seed)
    patterns.extend(tuple(rng.randrange(3) for _ in range(24)) for _ in range(20))
    return tuple(patterns)
