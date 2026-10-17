"""유전체·유전 연산·선발·행동 기록과 진행 보상을 재구성한다."""

import random
import numpy as np

from evosim_interaction_v02.audit import check_metrics
from evosim_interaction_v02.datasets import make_split
from .network import Genome, ModuleIds, initial_genome, validate_condition
from evosim_interaction_v02.rollout import derive_seed
from evosim_interaction_v02.storage import read_json, as_tuple, file_digest
from .model import NETWORK, Treatment, fitness_key, ranks_for, breed, ranking_diagnostics
from .evaluation import training_metrics, selection_key, trajectory_diagnostics
from .protocol import load, validation_episodes
from .search import directory, run_search


def verify_trajectories(records, lookup, *, training=False):
    """로그의 정수 행동만으로 환경을 batch 재생. scalar 기준은 별도 검사한다."""
    if not records:
        raise ValueError("No episode records")
    specs = [lookup[r["episode_id"]] for r in records]
    lengths = np.asarray([len(r["actions"]) for r in records])
    if np.any((lengths < 1) | (lengths > 24)):
        raise ValueError("Invalid episode length")
    actions = np.frombuffer(b"".join(r["actions"].encode("ascii").ljust(24, b"/") for r in records),
                            dtype=np.uint8).astype(np.int16).reshape(-1, 24) - 48
    rewards = np.asarray([r["rewards_cents"] + [0] * (24 - len(r["actions"])) for r in records])
    if training:
        if any(len(r["progress_units"]) != len(r["actions"]) for r in records):
            raise ValueError("Progress record length mismatch")
        progress = np.asarray([r["progress_units"] + [0] * (24 - len(r["actions"])) for r in records])
    position = np.asarray([ep.start for ep in specs])
    targets = np.asarray([ep.targets for ep in specs])
    count = np.zeros(len(records), dtype=int)
    walls = np.zeros(len(records), dtype=int)
    failed = np.zeros(len(records), dtype=int)
    minimum = abs(position - targets[:, 0])
    first_correct = np.zeros(len(records), dtype=bool)
    next_correct = next_total = 0
    for step in range(24):
        active = step < lengths
        action = actions[:, step]
        if np.any(active & ((action < 0) | (action > 2) | (count == 3))):
            raise ValueError("Invalid action or actions after success")
        goal = targets[np.arange(len(records)), np.minimum(count, 2)]
        correct = np.where(goal < position, 0, np.where(goal > position, 1, 2))
        if step == 0:
            first_correct = action == correct
        elif np.any(active):
            changed_goal = active & (rewards[:, step - 1] == 99)
            next_total += int(changed_goal.sum())
            next_correct += int((changed_goal & (action == correct)).sum())
        collected = active & (action == 2) & (position == goal)
        expected_reward = np.where(active, np.where(collected, 99, -1), 0)
        if not np.array_equal(rewards[:, step], expected_reward):
            raise ValueError("Reward differs from action replay")
        displacement = np.where(action == 0, -1, np.where(action == 1, 1, 0))
        new_position = np.clip(position + displacement, 0, 6)
        distance = abs(new_position - goal)
        expected_units = np.where(active & (action != 2), np.maximum(minimum - distance, 0), 0)
        if training and not np.array_equal(progress[:, step], expected_units):
            raise ValueError("Progress differs from action replay")
        minimum = np.where(active, np.minimum(minimum, distance), minimum)
        walls += active & (action != 2) & (position == new_position)
        failed += active & (action == 2) & ~collected
        position = np.where(active, new_position, position)
        count += collected
        next_goal = targets[np.arange(len(records)), np.minimum(count, 2)]
        minimum = np.where(collected & (count < 3), abs(position - next_goal), minimum)
        if np.any(active & (step == lengths - 1) & (count < 3) & (step != 23)):
            raise ValueError("Failure stopped before exhausting budget")
    for actual, key in ((count, "collected"), (position, "final_position"), (walls, "walls"), (failed, "failed_collect")):
        if not np.array_equal(actual, [r[key] for r in records]):
            raise ValueError("Outcome differs from action replay: " + key)
    if not np.array_equal(count == 3, [r["success"] for r in records]):
        raise ValueError("Success flag differs")
    if any(r["optimal_steps"] != ep.optimal_steps for r, ep in zip(records, specs)):
        raise ValueError("Optimal length differs")
    if training and (np.any(progress.sum(axis=1) > 18) or np.any(progress < 0)):
        raise ValueError("Progress bound violated")
    successes = count == 3
    wrong = ~first_correct
    failures = ~successes
    repeated = sum(not r["success"] and len(set(r["actions"][-6:])) == 1 for r in records)
    ratio = lambda a, b: a / b if b else None
    return {"episodes": len(records), "first_correct": int(first_correct.sum()),
            "first_correct_rate": float(first_correct.mean()), "first_wrong": int(wrong.sum()),
            "recovered_after_first_wrong": int((wrong & successes).sum()),
            "recovery_rate": ratio(int((wrong & successes).sum()), int(wrong.sum())),
            "goal_change_correct": next_correct, "goal_changes": next_total,
            "goal_change_correct_rate": ratio(next_correct, next_total),
            "failures": int(failures.sum()), "repeated_last_six_failures": repeated,
            "repeated_failure_rate": ratio(repeated, int(failures.sum()))}


def audit_job(root, condition, method, seed):
    config = load(root, condition)
    path = directory(root, condition, method, seed)
    state = read_json(path / "checkpoint.json")
    if state["next_round"] != config["rounds"]:
        raise ValueError("Audit cannot run incomplete search")
    result = run_search(root, condition, method, seed)
    treatment = Treatment("F0")
    split = make_split()
    lookup = {ep.episode_id: ep for ep in (*split.train, *split.validation)}
    rng = random.Random(derive_seed("q1007:v01:initial", seed))
    expected = [initial_genome(rng, NETWORK) for _ in range(config["population"])]
    if [g.to_dict() for g in expected] != result["initial_population"]:
        raise ValueError("Initial population differs")
    rng = random.Random(derive_seed("q1007:v01:search", seed, method))
    allocator = ModuleIds()
    ancestry = [{"kind": "initial", "parents": [], "mutation": None} for _ in expected]
    best = initial = final = None
    endpoints = {}
    genomes = calls = 0
    for generation in range(config["rounds"]):
        saved = read_json(path / f"round_{generation:03d}.json.gz")
        sampled = random.Random(derive_seed("q1007:v01:training_batch", seed, generation)).sample(list(split.train), 24)
        if saved["round"] != generation or saved["training_episodes"] != [e.episode_id for e in sampled]:
            raise ValueError("Training round or batch differs")
        rows = saved["candidates"]
        if len(rows) != config["population"] or [r["slot"] for r in rows] != list(range(len(rows))):
            raise ValueError("Candidate slots incomplete")
        verify_trajectories([r for row in rows for r in row["training_records"]], lookup, training=True)
        if generation in config["validation_rounds"]:
            verify_trajectories([r for row in rows for r in row["validation_records"]], lookup)
        population, keys, round_best = [], [], None
        for row, previous, parent in zip(rows, expected, ancestry):
            genome = Genome.from_dict(row["genome"], NETWORK)
            validate_condition(genome, "F0")
            if genome.digest != row["digest"] or genome.digest != previous.digest or row["ancestry"] != parent:
                raise ValueError("Genotype or genetic ancestry differs")
            check_metrics(row["training"], row["training_records"])
            metrics = training_metrics(row["training_records"], treatment)
            if any(row["training"][k] != v for k, v in metrics.items()):
                raise ValueError("Training score differs")
            if [(r["episode_id"], r["repeat"]) for r in row["training_records"]] != [(e.episode_id, 0) for e in sampled]:
                raise ValueError("Training record split/order differs")
            key = list(fitness_key(metrics, treatment))
            if row["fitness_key"] != key:
                raise ValueError("Breeding fitness key differs")
            population.append(genome)
            keys.append(key)
            calls += len(row["training_records"])
            if generation in config["validation_rounds"]:
                records, metrics = row["validation_records"], row["validation"]
                check_metrics(metrics, records)
                if [(r["episode_id"], r["repeat"]) for r in records] != [(e.episode_id, rep) for rep in range(3) for e in validation_episodes(config)]:
                    raise ValueError("Validation record split/order differs")
                calls += len(records)
                source = "validation"
            elif config["phase"] == "pilot" or (config["fixture"] and not config["validation_rounds"]):
                metrics, source = row["training"], "training_pilot"
            else:
                continue
            candidate = {"genome": genome.to_dict(), "digest": genome.digest, "metrics": metrics,
                         "round": generation, "slot": row["slot"], "selection_source": source}
            if round_best is None or selection_key(metrics) > selection_key(round_best["metrics"]):
                round_best = candidate
            if best is None or selection_key(metrics) > selection_key(best["metrics"]):
                best = candidate
        if [r["selection_rank"] for r in rows] != ranks_for(keys):
            raise ValueError("Ordinal breeding ranks differ")
        diagnostic = ranking_diagnostics([r["training"] for r in rows], treatment)
        if diagnostic != saved["ranking_diagnostics"]:
            raise ValueError("Ranking diagnostic differs")
        if str(generation + 1) in config["endpoint_rounds"]:
            endpoints[config["endpoint_rounds"][str(generation + 1)]] = {"candidate": best, "through_round": generation + 1, "logical_episode_slots": calls}
        history = {"round": generation, "training_mean_reward": sum(r["training"]["mean_reward"] for r in rows) / len(rows),
                   "training_mean_score": sum(r["training"]["mean_training_score"] for r in rows) / len(rows),
                   "training_mean_success": sum(r["training"]["success_rate"] for r in rows) / len(rows),
                   "best_training_reward": max(r["training"]["mean_reward"] for r in rows),
                   "best_training_score": max(r["training"]["mean_training_score"] for r in rows),
                   "validation_performed": generation in config["validation_rounds"],
                   "best_validation": best["metrics"] if best else None,
                   "mean_modules": sum(len(g.modules) for g in population) / len(rows),
                   "ranking": {key: value for key, value in diagnostic.items() if not key.endswith("order")}}
        if result["history"][generation] != history:
            raise ValueError("History differs from records")
        if generation == 0:
            initial = round_best
        if generation == config["rounds"] - 1:
            final = round_best
        else:
            if method == "evolution":
                expected, ancestry = breed(population, keys, rng, allocator, "F0")
            else:
                expected = [initial_genome(rng, NETWORK) for _ in population]
                ancestry = [{"kind": "independent_initialization", "parents": [], "mutation": None} for _ in population]
        genomes += len(population)
    if any(result[role] != value for role, value in (("best", best), ("initial", initial), ("final", final))):
        raise ValueError("Selected candidate differs from audited winner")
    if endpoints != result["endpoints"] or endpoints != state["endpoints"]:
        raise ValueError("끝점 후보가 해당 시점 누적 최선과 다름")
    if calls != result["logical_episode_slots"] or rng.getstate() != as_tuple(state["rng_state"]) or allocator.next_id != state["next_module_id"]:
        raise ValueError("Counters, RNG, or module IDs differ")
    if [Genome.restore(g, NETWORK).to_dict() for g in state["population"]] != [g.to_dict() for g in population]:
        raise ValueError("Final checkpoint population differs")
    return {"passed": True, "condition": condition, "method": method, "seed": seed,
            "genomes_checked": genomes, "episode_records_replayed": calls}

