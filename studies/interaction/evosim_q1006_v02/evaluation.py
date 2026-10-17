"""공통 환경 평가와 훈련에만 사용하는 제한된 진행 점수."""

import random
import numpy as np

from evosim_interaction_v02.environment import ACTION_BUDGET, LineCollectEnv
from evosim_interaction_v02.evaluation import summarize, select_actions, selection_key
from evosim_interaction_v02.policy import BatchNetwork
from evosim_interaction_v02.rollout import derive_seed
from .model import NETWORK


def replay_record(episode, record):
    """환경·진행 보상의 scalar 기준. 정책은 다시 실행하지 않는다."""
    env = LineCollectEnv()
    env.reset(episode)
    minimum = abs(env.state.position - env.state.goal)
    units, rewards, walls, failed, correct = [], [], 0, 0, []
    for action in record["actions"]:
        before = env.state
        expected = 0 if before.goal < before.position else 1 if before.goal > before.position else 2
        correct.append(int(action) == expected)
        step = env.step(int(action))
        after = env.state
        distance = abs(after.position - before.goal)
        bonus = max(0, minimum - distance) if int(action) != 2 else 0
        minimum = min(minimum, distance)
        if step.collected and not step.terminated:
            minimum = abs(after.position - after.goal)
        units.append(bonus)
        rewards.append(round(step.reward * 100))
        walls += step.wall_collision
        failed += step.failed_collect
    if not env.state.terminated or rewards != record["rewards_cents"]:
        raise ValueError("Trajectory reward/termination differs from reference environment")
    if (env.state.collected, env.state.position, walls, failed) != (record["collected"], record["final_position"], record["walls"], record["failed_collect"]):
        raise ValueError("Trajectory outcome differs from reference environment")
    if not 0 <= sum(units) <= 18:
        raise AssertionError("Progress bound violated")
    return units, correct


def training_metrics(records, treatment):
    metrics = summarize(records)
    total_units = sum(sum(record["progress_units"]) for record in records)
    metrics["mean_progress_units"] = total_units / len(records)
    metrics["mean_progress_bonus"] = total_units / (60 * len(records))
    # R1은 1/300 정수 단위로 합산해 수학적으로 같은 총점의 합산 순서 오차를 막는다.
    metrics["mean_training_score"] = ((3 * sum(sum(r["rewards_cents"]) for r in records) + 5 * total_units) / (300 * len(records))
                                       if treatment.shaping else metrics["mean_reward"])
    return metrics


def evaluate(genome, episodes, run_seed, purpose, treatment, *, repeats=1, training=False):
    if training != purpose.startswith("train:"):
        raise ValueError("Progress score is available only for explicit training purposes")
    episodes = tuple(episodes)
    if not episodes or type(repeats) is not int or repeats < 1:
        raise ValueError("Nonempty episodes and positive integer repeats required")
    genome.validate(NETWORK)
    before_digest = genome.digest
    rows = [(episode, rep) for rep in range(repeats) for episode in episodes]
    size = len(rows)
    rngs = [random.Random(derive_seed(purpose, run_seed, ep.episode_id, rep)) for ep, rep in rows]
    targets = np.asarray([ep.targets for ep, _ in rows], dtype=int)
    position = np.asarray([ep.start for ep, _ in rows], dtype=int)
    minimum = abs(position - targets[:, 0])
    collected = np.zeros(size, dtype=int)
    steps = np.zeros(size, dtype=int)
    walls = np.zeros(size, dtype=int)
    failed_collect = np.zeros(size, dtype=int)
    actions_log = np.full((size, ACTION_BUDGET), -1, dtype=np.int8)
    rewards_log = np.zeros((size, ACTION_BUDGET), dtype=np.int16)
    progress_log = np.zeros((size, ACTION_BUDGET), dtype=np.int8) if training else None
    policy = BatchNetwork(genome, size, NETWORK)
    ties = 0
    for step in range(ACTION_BUDGET):
        active = collected < 3
        if not active.any():
            break
        goal = targets[np.arange(size), np.minimum(collected, 2)]
        obs = np.column_stack((position / 6, goal / 6, (3 - collected) / 3,
                               np.full(size, (ACTION_BUDGET - step) / ACTION_BUDGET)))
        actions, step_ties = select_actions(policy.forward(obs), rngs, active)
        ties += step_ties
        did_collect = active & (actions == 2) & (position == goal)
        bad_collect = active & (actions == 2) & (position != goal)
        movement = np.where(actions == 0, -1, np.where(actions == 1, 1, 0))
        proposed = np.clip(position + movement, 0, 6)
        hit_wall = active & (actions != 2) & (proposed == position)
        if training:
            distance = abs(proposed - goal)
            progress_log[active, step] = np.where(actions != 2, np.maximum(0, minimum - distance), 0)[active]
            minimum = np.where(active, np.minimum(minimum, distance), minimum)
        position = np.where(active, proposed, position)
        collected += did_collect
        if training:
            next_goal = targets[np.arange(size), np.minimum(collected, 2)]
            minimum = np.where(did_collect & (collected < 3), abs(position - next_goal), minimum)
        steps += active
        walls += hit_wall
        failed_collect += bad_collect
        actions_log[active, step] = actions[active]
        rewards_log[active, step] = 100 * did_collect[active].astype(int) - 1
    records = []
    for row, (episode, rep) in enumerate(rows):
        length = int(steps[row])
        record = {"episode_id": episode.episode_id, "repeat": rep,
                  "actions": "".join(str(int(a)) for a in actions_log[row, :length]),
                  "rewards_cents": rewards_log[row, :length].tolist(),
                  "collected": int(collected[row]), "success": bool(collected[row] == 3),
                  "walls": int(walls[row]), "failed_collect": int(failed_collect[row]),
                  "optimal_steps": episode.optimal_steps, "final_position": int(position[row])}
        if training:
            record["progress_units"] = progress_log[row, :length].tolist()
        records.append(record)
    if before_digest != genome.digest:
        raise AssertionError("Evaluation mutated genome")
    metrics = training_metrics(records, treatment) if training else summarize(records)
    metrics["ties"] = ties
    return metrics, records


def trajectory_diagnostics(records, lookup):
    first_correct = first_wrong = recovered = goal_correct = goal_total = repeated = 0
    for row in records:
        _, correct = replay_record(lookup[row["episode_id"]], row)
        first_correct += correct[0]
        first_wrong += not correct[0]
        recovered += not correct[0] and row["success"]
        for index in range(1, len(correct)):
            if row["rewards_cents"][index - 1] == 99:
                goal_total += 1
                goal_correct += correct[index]
        repeated += not row["success"] and len(set(row["actions"][-6:])) == 1
    failures = sum(not row["success"] for row in records)
    ratio = lambda a, b: a / b if b else None
    return {"episodes": len(records), "first_correct": first_correct, "first_correct_rate": ratio(first_correct, len(records)),
            "first_wrong": first_wrong, "recovered_after_first_wrong": recovered, "recovery_rate": ratio(recovered, first_wrong),
            "goal_change_correct": goal_correct, "goal_changes": goal_total, "goal_change_correct_rate": ratio(goal_correct, goal_total),
            "failures": failures, "repeated_last_six_failures": repeated, "repeated_failure_rate": ratio(repeated, failures)}
