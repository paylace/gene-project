"""기존 환경·정책 선택을 유지한 batch 평가. 원래 평가 루프에서 관측 변환·진단만 추가."""
import random
import numpy as np
from evosim_interaction_v02.environment import ACTION_BUDGET
from evosim_interaction_v02.policy import BatchNetwork
from evosim_interaction_v02.rollout import derive_seed
from evosim_interaction_v02.evaluation import select_actions, summarize
from .model import SETTINGS, observations


def summarize_records(records):
    result = summarize(records)
    result.update(unproductive_tail_rate=sum(r["tail_kind"] in ("wall", "failed_collect") for r in records) / len(records),
        last_six_identical_rate=sum(r["tail_identical"] for r in records) / len(records),
        peak_abs_logit_mean=float(np.mean([r["peak_abs_logit"] for r in records])),
        peak_abs_logit_median=float(np.median([r["peak_abs_logit"] for r in records])),
        peak_abs_logit_p90=float(np.percentile([r["peak_abs_logit"] for r in records], 90, method="linear")),
        peak_abs_logit_max=max(r["peak_abs_logit"] for r in records))
    return result


def evaluate(genome, episodes, run_seed, purpose, condition, repeats=1):
    """No genome mutation and no learning objects. Columns have independent state."""
    episodes = tuple(episodes)
    if not episodes or repeats < 1:
        raise ValueError("Nonempty episodes and positive repeats required")
    settings = SETTINGS
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
    wall_log = np.zeros((size, ACTION_BUDGET), dtype=bool)
    failed_log = np.zeros((size, ACTION_BUDGET), dtype=bool)
    peaks = np.zeros(size)
    ties = 0
    for step in range(ACTION_BUDGET):
        active = collected < 3
        if not active.any():
            break
        goal = targets[np.arange(size), np.minimum(collected, 2)]
        obs = np.column_stack((position / 6, goal / 6, (3 - collected) / 3,
                               np.full(size, (ACTION_BUDGET - step) / ACTION_BUDGET)))
        logits = policy.forward(observations(obs, condition))
        peaks = np.where(active, np.maximum(peaks, np.abs(logits).max(axis=1)), peaks)
        actions, step_ties = select_actions(logits, rngs, active)
        ties += step_ties
        did_collect = active & (actions == 2) & (position == goal)
        bad_collect = active & (actions == 2) & (position != goal)
        movement = np.where(actions == 0, -1, np.where(actions == 1, 1, 0))
        proposed = np.clip(position + movement, 0, 6)
        hit_wall = active & (actions != 2) & (proposed == position)
        position = np.where(active, proposed, position)
        collected += did_collect
        steps += active
        wall_log[active, step] = hit_wall[active]
        failed_log[active, step] = bad_collect[active]
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
        tail = record["actions"][-6:]
        identical = length >= 6 and len(set(tail)) == 1
        tail_wall = identical and bool(wall_log[row, length-6:length].all())
        tail_failed = identical and bool(failed_log[row, length-6:length].all())
        record.update(tail_identical=identical, tail_kind="wall" if tail_wall else "failed_collect" if tail_failed else "other",
                      peak_abs_logit=float(peaks[row]))
        records.append(record)
    if before != genome.digest:
        raise AssertionError("Evaluation mutated genome")
    metrics = summarize_records(records)
    metrics["ties"] = ties
    return metrics, records

