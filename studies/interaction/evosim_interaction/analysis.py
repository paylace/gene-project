"""Predeclared seed-level decisions; no candidate selection on test results."""

import argparse
from pathlib import Path

import numpy as np

from .storage import read_json, write_json


def paired_decision(differences, seed=71004):
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) != 10 or not np.isfinite(values).all():
        raise ValueError("Ten completed seed differences are required")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    lower, upper = np.percentile(draws, [2.5, 97.5]).tolist()
    mean = float(values.mean())
    positive = int((values > 0).sum())
    return {"mean_difference": mean, "ci95": [lower, upper], "positive_seeds": positive,
            "passes_difference_rule": mean >= 0.10 and lower > 0 and positive >= 8}


def create_summary(rows, blind_rate):
    if len(rows) != 10 or len({row["seed"] for row in rows}) != 10:
        raise ValueError("Exactly ten unique seed pairs required")
    evo = np.asarray([row["evolution"]["success_rate"] for row in rows])
    random_search = np.asarray([row["random"]["success_rate"] for row in rows])
    capability = paired_decision(evo - blind_rate)
    search = paired_decision(evo - random_search)
    probes = [row["probe"]["both_correct_rate"] for row in rows]
    passing = sum(rate >= 0.8 for rate in probes)
    capability.update({"probe_mean": float(np.mean(probes)), "probe_passing_seeds": passing,
                       "q1_supported_by_declared_rule": capability["passes_difference_rule"] and passing >= 8})
    return {"rows": rows, "blind_success_rate": blind_rate, "capability": capability,
            "evolution_advantage": search,
            "evolution_mean": float(evo.mean()), "evolution_sd": float(evo.std(ddof=1)),
            "random_mean": float(random_search.mean()), "random_sd": float(random_search.std(ddof=1))}


def analyze(root):
    root = Path(root)
    config = read_json(root / "protocol.json")
    if config["pilot"]:
        raise ValueError("Pilot results cannot receive a confirmatory decision")
    baselines = read_json(root / "baseline_results.json")
    rows = []
    for seed in config["seeds"]:
        evolution = read_json(root / f"evolution_{seed}" / "test.json.gz")
        random_search = read_json(root / f"random_{seed}" / "test.json.gz")
        rows.append({"seed": seed, "evolution": evolution["best"]["metrics"],
                     "random": random_search["best"]["metrics"],
                     "initial": evolution["initial"]["metrics"],
                     "final": evolution["final"]["metrics"], "probe": evolution["probe"]})
    result = create_summary(rows, baselines["blind"]["test_metrics"]["success_rate"])
    result["uniform_random"] = baselines["uniform_random"]["metrics"]
    result["coordinate"] = baselines["coordinate"]["metrics"]
    result["initial_mean"] = float(np.mean([row["initial"]["success_rate"] for row in rows]))
    result["final_mean"] = float(np.mean([row["final"]["success_rate"] for row in rows]))
    write_json(root / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    print(analyze(args.root))
