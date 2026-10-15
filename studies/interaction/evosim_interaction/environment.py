"""일곱 칸 이동·수집 환경 v0.1. 정책에는 observe()의 네 값만 전달한다."""

from dataclasses import dataclass

WIDTH = 7
GOAL_COUNT = 3
ACTION_BUDGET = 24
ACTION_COST = 0.01
LEFT, RIGHT, COLLECT = range(3)
Observation = tuple[float, float, float, float]


def valid_action(action: object) -> bool:
    return type(action) is int and action in (LEFT, RIGHT, COLLECT)


@dataclass(frozen=True)
class EpisodeSpec:
    start: int
    targets: tuple[int, int, int]

    def __post_init__(self):
        if type(self.start) is not int or not 0 <= self.start < WIDTH:
            raise ValueError("초기 위치는 0~6의 정수여야 합니다.")
        if type(self.targets) is not tuple or len(self.targets) != GOAL_COUNT:
            raise ValueError("목표 목록은 정수 세 개의 튜플이어야 합니다.")
        previous = self.start
        for goal in self.targets:
            if type(goal) is not int or not 0 <= goal < WIDTH or goal == previous:
                raise ValueError("목표는 직전 수집 위치와 다른 0~6의 정수여야 합니다.")
            previous = goal

    @property
    def episode_id(self) -> str:
        return f"{self.start}:" + ",".join(map(str, self.targets))

    @property
    def optimal_steps(self) -> int:
        positions = (self.start,) + self.targets
        return GOAL_COUNT + sum(abs(b - a) for a, b in zip(positions, positions[1:]))


@dataclass(frozen=True)
class State:
    position: int
    goal: int | None
    collected: int
    budget: int
    terminated: bool


@dataclass(frozen=True)
class StepResult:
    observation: Observation | None
    reward: float
    terminated: bool
    collected: bool
    wall_collision: bool
    failed_collect: bool


class LineCollectEnv:
    """step()으로 상태를 갱신하며 미래 목표는 환경 내부에 보관한다."""

    def __init__(self):
        self._episode: EpisodeSpec | None = None
        self._state: State | None = None

    @property
    def state(self) -> State:
        if self._state is None:
            raise RuntimeError("reset()을 먼저 호출해야 합니다.")
        return self._state

    def reset(self, episode: EpisodeSpec) -> Observation:
        if not isinstance(episode, EpisodeSpec):
            raise TypeError("EpisodeSpec이 필요합니다.")
        self._episode = episode
        self._state = State(episode.start, episode.targets[0], 0, ACTION_BUDGET, False)
        return self.observe()

    def observe(self) -> Observation:
        state = self.state
        if state.terminated:
            raise RuntimeError("종료된 에피소드에는 다음 관측이 없습니다.")
        assert state.goal is not None
        return (
            state.position / (WIDTH - 1),
            state.goal / (WIDTH - 1),
            (GOAL_COUNT - state.collected) / GOAL_COUNT,
            state.budget / ACTION_BUDGET,
        )

    def step(self, action: int) -> StepResult:
        state = self.state
        if state.terminated:
            raise RuntimeError("종료 후 step()을 호출할 수 없습니다.")
        if not valid_action(action):
            raise ValueError("행동은 0(왼쪽), 1(오른쪽), 2(수집)의 정수여야 합니다.")
        position = state.position
        wall_collision = False
        collected = False
        failed_collect = False
        if action in (LEFT, RIGHT):
            proposed = position + (-1 if action == LEFT else 1)
            position = min(WIDTH - 1, max(0, proposed))
            wall_collision = position == state.position
        else:
            collected = position == state.goal
            failed_collect = not collected

        count = state.collected + int(collected)
        budget = state.budget - 1
        terminated = count == GOAL_COUNT or budget == 0
        goal = state.goal
        if terminated:
            goal = None
        elif collected:
            assert self._episode is not None
            goal = self._episode.targets[count]
        self._state = State(position, goal, count, budget, terminated)
        return StepResult(
            None if terminated else self.observe(),
            float(collected) - ACTION_COST,
            terminated,
            collected,
            wall_collision,
            failed_collect,
        )
