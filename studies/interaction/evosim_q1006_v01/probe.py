"""무작위 선택 대신 정확한 동률 집합으로 초기 반응을 측정한다."""
import numpy as np
from evosim_interaction_v02.policy import BatchNetwork
from .model import NETWORK


def correct(position, goal):
    return 2 if position == goal else (0 if goal < position else 1)


def choices(logits):
    logits = np.asarray(logits, dtype=float)
    exp = np.exp(np.clip(logits - logits.max(axis=1, keepdims=True), -10, 10))
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    maximum = probabilities.max(axis=1, keepdims=True)
    mask = abs(probabilities - maximum) <= np.maximum(1e-9, 1e-9 * np.maximum(abs(probabilities), abs(maximum)))
    return [np.flatnonzero(row).tolist() for row in mask]


def summarize_probe(records):
    lookup = {(r["position"], r["goal"], r["remaining_goals"], r["remaining_actions"]): r for r in records}
    if len(records) != 441 or len(lookup) != 441:
        raise ValueError("Complete unique cold states required")
    strict = 0
    for remaining in (1, 2, 3):
        for budget in (1, 12, 24):
            for position in range(1, 6):
                for left in range(position):
                    for right in range(position + 1, 7):
                        a, b = lookup[position, left, remaining, budget], lookup[position, right, remaining, budget]
                        strict += a["choices"] == [0] and b["choices"] == [1]
    collecting = [r for r in records if r["position"] == r["goal"]]
    probability = lambda r: (1 / len(r["choices"]) if correct(r["position"], r["goal"]) in r["choices"] else 0.0)
    return {"states": 441, "direction_pairs": 315, "strict_direction_pairs_correct": strict,
            "strict_direction_pair_rate": strict / 315,
            "collection_states": 63, "strict_collection_correct": sum(r["choices"] == [2] for r in collecting),
            "strict_collection_rate": sum(r["choices"] == [2] for r in collecting) / 63,
            "mean_correct_tie_probability": sum(probability(r) for r in records) / 441,
            "collection_correct_tie_probability": sum(probability(r) for r in collecting) / 63}


def probe(genome):
    states = [(p, g, n, b) for n in (1, 2, 3) for b in (1, 12, 24) for p in range(7) for g in range(7)]
    raw = [[p / 6, g / 6, n / 3, b / 24] for p, g, n, b in states]
    policy = BatchNetwork(genome, len(states), NETWORK)
    logits = policy.forward(np.asarray(raw,dtype=float))
    selected = choices(logits)
    records = [{"position": p, "goal": g, "remaining_goals": n, "remaining_actions": b,
                "logits": output.tolist(), "choices": choice}
               for (p, g, n, b), output, choice in zip(states, logits, selected)]
    return summarize_probe(records), records
