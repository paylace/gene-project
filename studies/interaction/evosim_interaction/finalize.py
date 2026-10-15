"""Lock all candidates first, then evaluate the held-out set and save state traces."""

import argparse
from dataclasses import asdict
from pathlib import Path

from .analysis import analyze
from .datasets import make_split
from .environment import LineCollectEnv
from .evaluation import evaluate, evaluate_policy, observation_probe, selection_key
from .network import Genome
from .policies import CoordinatePolicy, FixedSequencePolicy, UniformRandomPolicy, blind_patterns
from .storage import manifest_digest, read_json, source_manifest, write_json


def reconstruct_states(episodes, records):
    lookup = {episode.episode_id: episode for episode in episodes}
    for record in records:
        env = LineCollectEnv()
        env.reset(lookup[record["episode_id"]])
        states = [asdict(env.state)]
        rewards, walls, failed = [], 0, 0
        for action in record["actions"]:
            result = env.step(int(action))
            rewards.append(round(result.reward * 100))
            walls += result.wall_collision
            failed += result.failed_collect
            states.append(asdict(env.state))
        if not env.state.terminated or rewards != record["rewards_cents"]:
            raise AssertionError("Recorded actions/rewards disagree with reference environment")
        if (env.state.collected != record["collected"] or walls != record["walls"]
                or failed != record["failed_collect"] or (env.state.collected == 3) != record["success"]):
            raise AssertionError("Recorded outcome disagrees with reference environment")
        record["states"] = states


def finalize(root):
    root = Path(root)
    config = read_json(root / "protocol.json")
    if config["pilot"]:
        raise ValueError("Do not expose test data in a pilot")
    if config["source_digest"] != manifest_digest(source_manifest()):
        raise ValueError("Source changed after protocol lock")
    split = make_split()
    locked = {}
    for seed in config["seeds"]:
        for method in config["methods"]:
            directory = root / f"{method}_{seed}"
            checkpoint = read_json(directory / "checkpoint.json")
            if checkpoint["next_round"] != config["rounds"]:
                raise ValueError("Every search must finish before any test evaluation")
            result = read_json(directory / "result.json")
            locked[f"{method}_{seed}"] = {role: result[role]["digest"] for role in ("best", "initial", "final")}
    patterns = blind_patterns()
    baseline_candidates = []
    selected = None
    for index, pattern in enumerate(patterns):
        metrics, records = evaluate_policy(FixedSequencePolicy(pattern), split.validation, 71002, "blind_validation")
        baseline_candidates.append({"index": index, "pattern": pattern, "metrics": metrics, "records": records})
        if selected is None or selection_key(metrics) > selection_key(selected["metrics"]):
            selected = baseline_candidates[-1]
    locked["blind_pattern_index"] = selected["index"]
    lock_path = root / "candidate_lock.json"
    if lock_path.exists() and read_json(lock_path) != locked:
        raise ValueError("Candidate selection changed after test lock")
    write_json(root / "blind_validation.json.gz", baseline_candidates)
    write_json(lock_path, locked)
    # No test scores are computed before the lock above.
    for seed in config["seeds"]:
        for method in config["methods"]:
            directory = root / f"{method}_{seed}"
            if (directory / "test.json.gz").exists():
                continue
            result = read_json(directory / "result.json")
            tested = {}
            for role in ("best", "initial", "final"):
                genome = Genome.from_dict(result[role]["genome"])
                metrics, records = evaluate(genome, split.test, seed, "test", repeats=3)
                reconstruct_states(split.test, records)
                tested[role] = {"digest": genome.digest, "metrics": metrics, "records": records}
            tested["probe"] = observation_probe(Genome.from_dict(result["best"]["genome"]), seed)
            write_json(directory / "test.json.gz", tested)
            print(f"TEST {method} {seed}: {tested['best']['metrics']['success_rate']:.4f}", flush=True)
    blind_metrics, blind_records = evaluate_policy(FixedSequencePolicy(tuple(selected["pattern"])), split.test, 71002, "blind_test")
    uniform_metrics, uniform_records = evaluate_policy(UniformRandomPolicy(), split.test, 71003, "uniform_test", repeats=20)
    coordinate_metrics, coordinate_records = evaluate_policy(CoordinatePolicy(), split.test, 71003, "coordinate_test")
    for records in (blind_records, uniform_records, coordinate_records):
        reconstruct_states(split.test, records)
    write_json(root / "baseline_records.json.gz", {"blind": blind_records, "uniform_random": uniform_records,
                                                  "coordinate": coordinate_records})
    write_json(root / "baseline_results.json", {
        "blind": {"pattern_index": selected["index"], "pattern": selected["pattern"],
                  "validation_metrics": selected["metrics"], "test_metrics": blind_metrics},
        "uniform_random": {"metrics": uniform_metrics}, "coordinate": {"metrics": coordinate_metrics},
    })
    summary = analyze(root)
    print({key: summary[key] for key in ("evolution_mean", "random_mean", "capability", "evolution_advantage")}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    finalize(parser.parse_args().root)
