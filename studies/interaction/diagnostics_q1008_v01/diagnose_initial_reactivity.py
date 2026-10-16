"""Q1-008 사후 진단: 동결 v02 초기 유전체의 반응성·고착·출력 누적과 선형 해의 존재 확인.

동결 소스(evosim_interaction_v02)는 읽기만 한다. 진화·선발·테스트를 실행하지 않는다.
작업 디렉터리 studies/interaction/ 에서 실행:
    python -B -X utf8 diagnostics_q1008_v01/diagnose_initial_reactivity.py
"""

import hashlib
import itertools
import json
import platform
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY))

from evosim_interaction_v02.environment import LineCollectEnv, EpisodeSpec  # noqa: E402
from evosim_interaction_v02.network import Genome, Module, initial_genome  # noqa: E402
from evosim_interaction_v02.policy import NetworkPolicy  # noqa: E402
from evosim_interaction_v02.settings import Settings  # noqa: E402

GENOME_SEEDS = range(1000, 1300)        # 조건별 초기 유전체 300개
EPISODE_SAMPLE_SEED = 0
EPISODES_PER_GENOME = 40
TIE_RNG_SEED = 0
FROZEN_FILES = ("environment.py", "network.py", "policy.py", "settings.py")


def all_episodes():
    result = []
    for start in range(7):
        for targets in itertools.product(range(7), repeat=3):
            previous, ok = start, True
            for goal in targets:
                if goal == previous:
                    ok = False
                    break
                previous = goal
            if ok:
                result.append(EpisodeSpec(start, targets))
    return result


def correct_action(position, goal):
    return 2 if position == goal else (0 if goal < position else 1)


def argmax_action(policy, observation, rng):
    return policy.act(observation, rng)


def first_step(policy):
    """같은 위치에서 목표만 바꿀 때의 첫 행동. 초기 내부 상태에서 계산한다."""
    rng = random.Random(TIE_RNG_SEED)
    table = {}
    for position in range(7):
        for goal in range(7):
            if goal == position:
                continue
            policy.reset()
            table[(position, goal)] = argmax_action(policy, (position / 6, goal / 6, 1.0, 1.0), rng)
    correct = sum(a == correct_action(p, g) for (p, g), a in table.items())
    responsive = any(len({table[(p, g)] for g in range(7) if g != p}) > 1 for p in range(7))
    adjacent = flips = 0
    for (p, g), a in table.items():
        for h in (g - 1, g + 1):
            if 0 <= h < 7 and h != p:
                adjacent += 1
                flips += table[(p, h)] != a
    return correct, len(table), responsive, flips, adjacent


def rollout(policy, spec):
    env = LineCollectEnv()
    observation = env.reset(spec)
    policy.reset()
    rng = random.Random(TIE_RNG_SEED)
    history, peak, correct, steps = [], 0.0, 0, 0
    while True:
        state = env.state
        logits = policy.forward(observation)
        peak = max(peak, max(abs(v) for v in logits))
        # act()와 같은 결정 규칙을 쓰되 forward를 두 번 호출하지 않도록 직접 계산한다.
        best = max(logits)
        choices = [i for i, v in enumerate(logits) if abs(v - best) <= 1e-9 * max(1.0, abs(best))]
        action = choices[0] if len(choices) == 1 else rng.choice(choices)
        correct += action == correct_action(state.position, state.goal)
        steps += 1
        result = env.step(action)
        history.append((action, result.failed_collect, result.wall_collision))
        if result.terminated:
            return history, peak, correct, steps, env.state.collected
        observation = result.observation


def diagnose(condition, sample):
    settings = Settings(condition)
    totals = dict(first_correct=0, first_total=0, responsive=0, flips=0, adjacent=0,
                  step_correct=0, steps=0, episodes=0, completed=0, stuck=0,
                  stuck_failed_collect=0, stuck_wall=0, stuck_other=0, output_self_rec_over_1=0, outputs=0)
    peaks = []
    for seed in GENOME_SEEDS:
        genome = initial_genome(random.Random(seed), settings)
        for out in ("out_0", "out_1", "out_2"):
            totals["outputs"] += 1
            totals["output_self_rec_over_1"] += genome.rec.get(out, {}).get(out, 0.0) > 1.0
        policy = NetworkPolicy(genome, settings)
        c, n, responsive, flips, adjacent = first_step(policy)
        totals["first_correct"] += c
        totals["first_total"] += n
        totals["responsive"] += responsive
        totals["flips"] += flips
        totals["adjacent"] += adjacent
        for spec in sample:
            history, peak, correct, steps, collected = rollout(policy, spec)
            peaks.append(peak)
            totals["step_correct"] += correct
            totals["steps"] += steps
            totals["episodes"] += 1
            totals["completed"] += collected == 3
            tail = history[-6:]
            if len(tail) == 6 and len({a for a, _, _ in tail}) == 1:
                totals["stuck"] += 1
                if all(f for _, f, _ in tail):
                    totals["stuck_failed_collect"] += 1
                elif all(w for _, _, w in tail):
                    totals["stuck_wall"] += 1
                else:
                    totals["stuck_other"] += 1
    peaks.sort()
    n_genomes = len(GENOME_SEEDS)
    return {
        "condition": condition,
        "genomes": n_genomes,
        "counts": totals,
        "first_action_correct_rate": totals["first_correct"] / totals["first_total"],
        "goal_responsive_genome_rate": totals["responsive"] / n_genomes,
        "adjacent_goal_change_flip_rate": totals["flips"] / totals["adjacent"],
        "all_step_correct_rate": totals["step_correct"] / totals["steps"],
        "completion_rate": totals["completed"] / totals["episodes"],
        "last6_identical_rate": totals["stuck"] / totals["episodes"],
        "stuck_breakdown_share": {
            "failed_collect": totals["stuck_failed_collect"] / max(1, totals["stuck"]),
            "wall_collision": totals["stuck_wall"] / max(1, totals["stuck"]),
            "other_moving": totals["stuck_other"] / max(1, totals["stuck"]),
        },
        "output_self_rec_weight_over_1_share": totals["output_self_rec_over_1"] / totals["outputs"],
        "episode_peak_abs_logit": {"median": statistics.median(peaks),
                                   "p90": peaks[int(len(peaks) * 0.9)], "max": peaks[-1]},
    }


def linear_solution_check(episodes):
    """동결 정책 코드로 표현한 선형 반응 해: 왼=k(p-g), 오=k(g-p), 수집=상수 c, 0<c<k/6."""
    settings = Settings("A2")
    k = 6.0
    genome = Genome(
        modules={"m0": Module("linear", 0.0, 0.5)},          # 입력 없음 → sigmoid(0)=0.5 상수
        ff={"out_0": {"in_0": k, "in_1": -k},
            "out_1": {"in_0": -k, "in_1": k},
            "out_2": {"m0": 1.0}},
        rec={}, settings=settings)
    genome.validate(settings)
    policy = NetworkPolicy(genome, settings)
    completed = 0
    optimal = 0
    for spec in episodes:
        env = LineCollectEnv()
        observation = env.reset(spec)
        policy.reset()
        rng = random.Random(TIE_RNG_SEED)
        steps = 0
        while True:
            result = env.step(policy.act(observation, rng))
            steps += 1
            if result.terminated:
                break
            observation = result.observation
        completed += env.state.collected == 3
        optimal += env.state.collected == 3 and steps == spec.optimal_steps
    return {"genome": genome.to_dict(), "episodes": len(episodes), "completed": completed,
            "completed_in_optimal_steps": optimal, "note": "수작업 구성. 진화로 발견한 성적이 아님"}


def main():
    episodes = all_episodes()
    sample = random.Random(EPISODE_SAMPLE_SEED).sample(episodes, 200)[:EPISODES_PER_GENOME]
    frozen = STUDY / "evosim_interaction_v02"
    result = {
        "diagnostic": "Q1-008 initial genome reactivity (post-hoc, exploratory)",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "frozen_source_sha256": {name: hashlib.sha256((frozen / name).read_bytes()).hexdigest()
                                 for name in FROZEN_FILES},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parameters": {"genome_seeds": [GENOME_SEEDS.start, GENOME_SEEDS.stop - 1],
                       "episode_sample_seed": EPISODE_SAMPLE_SEED,
                       "episodes_per_genome": EPISODES_PER_GENOME,
                       "episode_ids": [spec.episode_id for spec in sample],
                       "decision": "argmax of logits, ties by Random(0) as in NetworkPolicy.act",
                       "total_episode_space": len(episodes)},
        "conditions": [diagnose(c, sample) for c in ("A0", "A1", "A2")],
        "linear_solution": linear_solution_check(episodes),
    }
    out = HERE / "results.json"
    if out.exists():
        raise SystemExit(f"기존 결과를 덮어쓰지 않습니다: {out}")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in result["conditions"]:
        print(json.dumps({k: v for k, v in row.items() if k != "counts"}, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result["linear_solution"].items() if k != "genome"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
