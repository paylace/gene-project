"""정책에는 관측만 제공하고, 환경 상태는 별도 실행 기록에 보존한다."""

import hashlib
import random
from dataclasses import dataclass

from .environment import EpisodeSpec, GOAL_COUNT, LineCollectEnv, Observation, State
from .policies import Policy


def derive_seed(*parts: str | int) -> int:
    # 자료형을 포함해 ["1", 2]와 [1, "2"]도 구별한다.
    import json

    encoded = json.dumps(parts, ensure_ascii=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(encoded.encode("ascii")).digest(), "big")


@dataclass(frozen=True)
class Transition:
    before: State
    observation: Observation
    action: int
    reward: float
    after: State
    collected: bool
    wall_collision: bool
    failed_collect: bool


@dataclass(frozen=True)
class EpisodeRecord:
    episode: EpisodeSpec
    transitions: tuple[Transition, ...]

    @property
    def success(self) -> bool:
        return self.transitions[-1].after.collected == GOAL_COUNT

    @property
    def total_reward(self) -> float:
        return sum(transition.reward for transition in self.transitions)

    @property
    def steps(self) -> int:
        return len(self.transitions)


def run_episode(policy: Policy, episode: EpisodeSpec, seed: int) -> EpisodeRecord:
    env = LineCollectEnv()
    observation = env.reset(episode)
    policy.reset()
    rng = random.Random(seed)
    transitions = []
    while True:
        before = env.state
        action = policy.act(observation, rng)
        result = env.step(action)
        transitions.append(Transition(
            before, observation, action, result.reward, env.state,
            result.collected, result.wall_collision, result.failed_collect,
        ))
        if result.terminated:
            break
        assert result.observation is not None
        observation = result.observation
    return EpisodeRecord(episode, tuple(transitions))
