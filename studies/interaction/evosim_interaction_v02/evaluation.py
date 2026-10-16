"""Frozen evaluation, raw episode records, and observation-pair diagnostics."""

import random
import numpy as np

from .environment import ACTION_BUDGET, EpisodeSpec
from .policy import BatchNetwork, NetworkPolicy
from .rollout import derive_seed, run_episode
from .settings import digest_json


def select_actions(logits, rngs, active):
    """v0.1과 동일한 batch 동률 처리. 관찰자도 같은 함수를 사용한다."""
    exps = np.exp(np.clip(logits - logits.max(axis=1, keepdims=True), -10.0, 10.0))
    probs = exps / exps.sum(axis=1, keepdims=True)
    maxima = probs.max(axis=1, keepdims=True)
    close = np.abs(probs - maxima) <= np.maximum(1e-9, 1e-9 * np.maximum(np.abs(probs), np.abs(maxima)))
    actions = np.argmax(probs, axis=1)
    ties = 0
    for row in np.flatnonzero(active & (close.sum(axis=1) > 1)):
        actions[row] = rngs[row].choice(np.flatnonzero(close[row]).tolist())
        ties += 1
    return actions, ties


def observe_step(genome, settings, previous, observation, logits, action, rng_state):
    x, goal = (int(round(value * 6)) for value in observation[:2])
    correct = 0 if goal < x else 1 if goal > x else 2
    ranked = sorted(float(value) for value in logits)
    record = {"observation": observation.tolist(), "previous_state_digest": digest_json(previous.tolist()),
              "rng_state_digest": digest_json(rng_state), "logits": logits.tolist(),
              "action": int(action), "correct_action": correct, "top_gap": ranked[-1] - ranked[-2],
              "correct_minus_selected": float(logits[correct] - logits[action]), "mirror": None}
    reflected = 2 * x - goal
    if goal != x and 0 <= reflected <= 6:
        pair_actions = []
        for target in (goal, reflected):
            cloned = BatchNetwork(genome, 1, settings)
            cloned.previous[:, 0] = previous
            altered = observation.copy()
            altered[1] = target / 6
            cloned_rng = random.Random(0)
            cloned_rng.setstate(rng_state)
            predicted, _ = select_actions(cloned.forward(altered[None, :]), [cloned_rng], np.array([True]))
            pair_actions.append(int(predicted[0]))
        if pair_actions[0] != action:
            raise AssertionError("Counterfactual original branch disagrees with live batch action")
        record["mirror"] = {"reflected_goal": reflected, "actions": pair_actions,
                            "both_correct": pair_actions == [correct, 1 - correct],
                            "action_switched": pair_actions[0] != pair_actions[1]}
    return record


def summarize(records):
    if not records:
        raise ValueError("Empty evaluation")
    count = len(records)
    successes = [record for record in records if record["success"]]
    return {
        "episodes": count,
        "success_rate": sum(r["success"] for r in records) / count,
        "mean_reward": sum(sum(r["rewards_cents"]) / 100 for r in records) / count,
        "mean_collected": sum(r["collected"] for r in records) / count,
        "mean_steps": sum(len(r["actions"]) for r in records) / count,
        "mean_walls": sum(r["walls"] for r in records) / count,
        "mean_failed_collect": sum(r["failed_collect"] for r in records) / count,
        "success_extra_steps": (sum(len(r["actions"]) - r["optimal_steps"] for r in successes) / len(successes)
                                if successes else None),
    }


def selection_key(metrics):
    return metrics["success_rate"], metrics["mean_reward"]


def evaluate(genome, episodes, run_seed, purpose, settings, repeats=1, observe=False):
    """No genome mutation and no learning objects. Columns have independent state."""
    episodes = tuple(episodes)
    if not episodes or repeats < 1:
        raise ValueError("Nonempty episodes and positive repeats required")
    genome.validate(settings)
    before = genome.digest
    rows = [(episode, rep) for rep in range(repeats) for episode in episodes]
    size = len(rows)
    rngs = [random.Random(derive_seed(purpose, run_seed, ep.episode_id, rep)) for ep, rep in rows]
    targets = np.asarray([ep.targets for ep, _ in rows], dtype=int)
    position = np.asarray([ep.start for ep, _ in rows], dtype=int)
    collected = np.zeros(size, dtype=int)
    steps = np.zeros(size, dtype=int)
    walls = np.zeros(size, dtype=int)
    failed_collect = np.zeros(size, dtype=int)
    actions_log = np.full((size, ACTION_BUDGET), -1, dtype=np.int8)
    rewards_log = np.zeros((size, ACTION_BUDGET), dtype=np.int16)
    policy = BatchNetwork(genome, size, settings)
    observations_log = [[] for _ in rows] if observe else None
    ties = 0
    for step in range(ACTION_BUDGET):
        active = collected < 3
        if not active.any():
            break
        goal = targets[np.arange(size), np.minimum(collected, 2)]
        obs = np.column_stack((position / 6, goal / 6, (3 - collected) / 3,
                               np.full(size, (ACTION_BUDGET - step) / ACTION_BUDGET)))
        previous = policy.previous.copy() if observe else None
        rng_states = [rng.getstate() for rng in rngs] if observe else None
        logits = policy.forward(obs)
        actions, step_ties = select_actions(logits, rngs, active)
        ties += step_ties
        if observe:
            for row in np.flatnonzero(active):
                observations_log[row].append(observe_step(genome, settings, previous[:, row], obs[row],
                                                          logits[row], actions[row], rng_states[row]))
        did_collect = active & (actions == 2) & (position == goal)
        bad_collect = active & (actions == 2) & (position != goal)
        movement = np.where(actions == 0, -1, np.where(actions == 1, 1, 0))
        proposed = np.clip(position + movement, 0, 6)
        hit_wall = active & (actions != 2) & (proposed == position)
        position = np.where(active, proposed, position)
        collected += did_collect
        steps += active
        walls += hit_wall
        failed_collect += bad_collect
        actions_log[active, step] = actions[active]
        rewards_log[active, step] = 100 * did_collect[active].astype(int) - 1
    records = []
    for row, (episode, rep) in enumerate(rows):
        length = int(steps[row])
        record = {
            "episode_id": episode.episode_id, "repeat": rep,
            "actions": "".join(str(int(a)) for a in actions_log[row, :length]),
            "rewards_cents": rewards_log[row, :length].tolist(),
            "collected": int(collected[row]), "success": bool(collected[row] == 3),
            "walls": int(walls[row]), "failed_collect": int(failed_collect[row]),
            "optimal_steps": episode.optimal_steps, "final_position": int(position[row]),
        }
        if observe:
            for index, logged in enumerate(observations_log[row]):
                logged.update({"reward_cents": record["rewards_cents"][index],
                               "terminated": index == length - 1})
            record["observations"] = observations_log[row]
        records.append(record)
    if before != genome.digest:
        raise AssertionError("Evaluation mutated genome")
    metrics = summarize(records)
    metrics["ties"] = ties
    return metrics, records


def evaluate_policy(policy, episodes, run_seed, purpose, repeats=1):
    records = []
    for rep in range(repeats):
        for episode in episodes:
            trajectory = run_episode(policy, episode, derive_seed(purpose, run_seed, episode.episode_id, rep))
            records.append({
                "episode_id": episode.episode_id, "repeat": rep,
                "actions": "".join(str(t.action) for t in trajectory.transitions),
                "rewards_cents": [round(t.reward * 100) for t in trajectory.transitions],
                "collected": trajectory.transitions[-1].after.collected,
                "success": trajectory.success,
                "walls": sum(t.wall_collision for t in trajectory.transitions),
                "failed_collect": sum(t.failed_collect for t in trajectory.transitions),
                "optimal_steps": episode.optimal_steps,
                "final_position": trajectory.transitions[-1].after.position,
            })
    return summarize(records), records


def observation_probe(genome, run_seed, settings, repeats=3):
    records = []
    for rep in range(repeats):
        for position in range(1, 6):
            for left in range(position):
                for right in range(position + 1, 7):
                    seed = derive_seed("probe", run_seed, rep, position, left, right)
                    left_policy, right_policy = NetworkPolicy(genome, settings), NetworkPolicy(genome, settings)
                    a_left = left_policy.act((position / 6, left / 6, 1.0, 1.0), random.Random(seed))
                    a_right = right_policy.act((position / 6, right / 6, 1.0, 1.0), random.Random(seed))
                    records.append({"repeat": rep, "position": position, "left": left, "right": right,
                                    "left_action": a_left, "right_action": a_right,
                                    "both_correct": a_left == 0 and a_right == 1})
    return {"both_correct_rate": sum(r["both_correct"] for r in records) / len(records), "records": records}
