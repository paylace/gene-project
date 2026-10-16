"""목표 목록을 단위로 나누고 좌우 반전의 균형을 보존하는 고정 분할."""

import hashlib
import json
import random
from dataclasses import dataclass
from itertools import product

from .environment import EpisodeSpec, WIDTH

SPLIT_SEED = 71001
GoalList = tuple[int, int, int]


def reflect(goals: GoalList) -> GoalList:
    return tuple(WIDTH - 1 - goal for goal in goals)


def goal_lists() -> tuple[GoalList, ...]:
    return tuple(
        goals
        for goals in product(range(WIDTH), repeat=3)
        if goals[0] != goals[1] and goals[1] != goals[2]
    )


def episodes_for(goals: tuple[GoalList, ...]) -> tuple[EpisodeSpec, ...]:
    return tuple(
        EpisodeSpec(start, targets)
        for targets in sorted(goals)
        for start in range(WIDTH)
        if start != targets[0]
    )


def all_episodes() -> tuple[EpisodeSpec, ...]:
    return episodes_for(goal_lists())


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[EpisodeSpec, ...]
    validation: tuple[EpisodeSpec, ...]
    test: tuple[EpisodeSpec, ...]

    def manifest(self) -> dict:
        return {
            name: [episode.episode_id for episode in getattr(self, name)]
            for name in ("train", "validation", "test")
        }

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.manifest(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def make_split(seed: int = SPLIT_SEED) -> DatasetSplit:
    representatives = sorted({min(goals, reflect(goals)) for goals in goal_lists()})
    random.Random(seed).shuffle(representatives)

    def expand(reps):
        goals = tuple(goal for representative in reps for goal in (representative, reflect(representative)))
        return episodes_for(goals)

    return DatasetSplit(
        expand(representatives[:63]),
        expand(representatives[63:84]),
        expand(representatives[84:]),
    )
