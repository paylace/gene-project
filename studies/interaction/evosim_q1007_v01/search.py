"""Q1-007 끝점별 후보와 훈련 점수·선발의 분리 탐색과 원자 재개. 시험은 잠금 뒤에만 허용한다."""

from pathlib import Path
import random
import time
from datetime import datetime, timezone
import uuid

from evosim_interaction_v02.datasets import make_split
from .evaluation import evaluate, selection_key
from .model import Treatment, NETWORK, fitness_key, ranks_for, breed, ranking_diagnostics
from .network import Genome, ModuleIds, initial_genome, validate_condition
from .protocol import check_access, load, validation_episodes
from evosim_interaction_v02.rollout import derive_seed
from evosim_interaction_v02.settings import digest_json
from evosim_interaction_v02.storage import as_tuple, file_digest, read_json, write_json, write_once


def directory(root, condition, method, seed):
    return Path(root) / condition / f"{method}_{seed}"


def attempt_accounting(path):
    known, interrupted, upper, forwards, forward_upper = 0, 0, 0, 0, 0
    for file in (Path(path) / "attempts").glob("*.json"):
        item = read_json(file)
        if item["status"] in ("evaluated", "committed"):
            known += item["completed_episode_calls"]
            forwards += item.get("completed_single_step_actions", 0)
        else:
            interrupted += 1
            upper += item["maximum_episode_calls"]
            forward_upper += item.get("maximum_single_step_actions", 0)
    return {"known_completed_episode_calls": known, "interrupted_attempts": interrupted,
            "unknown_interrupted_calls_upper_bound": upper,
            "known_single_step_actions": forwards, "unknown_single_step_actions_upper_bound": forward_upper}


def validate_checkpoint(state):
    body = {key: value for key, value in state.items() if key != "checkpoint_digest"}
    if state.get("checkpoint_digest") != digest_json(body):
        raise ValueError("Checkpoint checksum mismatch")


def run_search(root, condition, method, seed, *, pause_after=None):
    started = time.perf_counter()
    root = Path(root)
    config = load(root, condition)
    if {"condition": condition, "method": method, "seed": seed} not in config["jobs"]:
        raise ValueError("Job outside protocol")
    treatment = Treatment.from_dict(config["settings"][condition])
    settings = NETWORK
    rounds, size = config["rounds"], config["population"]
    path = directory(root, condition, method, seed)
    identity = {"experiment_id": config["experiment_id"], "phase": config["phase"],
                "protocol_digest": config["protocol_digest"], "source_digest": config["source_digest"],
                "settings_digest": treatment.digest, "network_settings_digest": settings.digest, "datasets_digest": config["datasets_digest"],
                "condition": condition, "method": method, "seed": seed, "rounds": rounds, "population": size}
    checkpoint = path / "checkpoint.json"
    if checkpoint.exists():
        state = read_json(checkpoint)
        if state["identity"] != identity:
            raise ValueError("Checkpoint identity differs")
        validate_checkpoint(state)
        if type(state["next_round"]) is not int or not 1 <= state["next_round"] <= rounds:
            raise ValueError("Checkpoint next round is invalid")
        if set(state["record_hashes"]) != {f"round_{g:03d}.json.gz" for g in range(state["next_round"])}:
            raise ValueError("Checkpoint round manifest incomplete")
        population = [Genome.restore(item, settings) for item in state["population"]]
        if len(population) != size:
            raise ValueError("Checkpoint population size mismatch")
        allocator = ModuleIds(state["next_module_id"])
        allocator.validate(population)
        for name, digest in state["record_hashes"].items():
            if file_digest(path / name) != digest:
                raise ValueError("Completed round changed or missing")
        if state["next_round"] == rounds:
            result = read_json(path / "result.json")
            if state["result_digest"] != digest_json(result):
                raise ValueError("Completed result changed")
            return result
        rng = random.Random(0)
        rng.setstate(as_tuple(state["rng_state"]))
        best, initial, initial_population = state["best"], state["initial"], state["initial_population"]
        endpoints = state["endpoints"]
        ancestry, history = state["ancestry"], state["history"]
        hashes, start, elapsed_before = state["record_hashes"], state["next_round"], state["elapsed_seconds"]
        episode_calls = state["logical_episode_slots"]
    else:
        initial_rng = random.Random(derive_seed("q1007:v01:initial", seed))
        population = [initial_genome(initial_rng, settings) for _ in range(size)]
        initial_population = [g.to_dict() for g in population]
        rng = random.Random(derive_seed("q1007:v01:search", seed, method))
        allocator = ModuleIds()
        best = initial = None
        endpoints = {}
        ancestry = [{"kind": "initial", "parents": [], "mutation": None} for _ in population]
        history, hashes = [], {}
        start, elapsed_before, episode_calls = 0, 0.0, 0
    split = make_split()
    for generation in range(start, rounds):
        calls_before = episode_calls
        attempt_path = path / "attempts" / f"{generation:03d}_{uuid.uuid4().hex}.json"
        attempt = {"round": generation, "status": "started", "started_utc": datetime.now(timezone.utc).isoformat(),
                   "maximum_single_step_actions": size * 24 * (24 + (len(validation_episodes(config)) * config["repeats"] if generation in config["validation_rounds"] else 0)),
                   "maximum_episode_calls": size * (24 + (len(validation_episodes(config)) * config["repeats"] if generation in config["validation_rounds"] else 0))}
        write_json(attempt_path, attempt)
        sampled = random.Random(derive_seed("q1007:v01:training_batch", seed, generation)).sample(list(split.train), 24)
        check_access(config, f"train:{generation}", sampled)
        rows, scores = [], []
        for slot, genome in enumerate(population):
            validate_condition(genome, "F0")
            metrics, records = evaluate(genome, sampled, seed, f"train:{generation}:q1007:v01:slot{slot}", treatment, training=True)
            episode_calls += len(records)
            scores.append(metrics["mean_training_score"])
            rows.append({"slot": slot, "genome": genome.to_dict(), "digest": genome.digest,
                         "ancestry": ancestry[slot], "training": metrics, "training_records": records,
                         "fitness_key": list(fitness_key(metrics, treatment))})
        selection_ranks = ranks_for([row["fitness_key"] for row in rows])
        for row, rank in zip(rows, selection_ranks):
            row["selection_rank"] = rank
        rank_diagnostic = ranking_diagnostics([row["training"] for row in rows], treatment)
        validation = generation in config["validation_rounds"]
        final_candidate = None
        if validation or config["phase"] == "pilot" or (config["fixture"] and not config["validation_rounds"]):
            for slot, genome in enumerate(population):
                if validation:
                    check_access(config, "validation", validation_episodes(config))
                    metrics, records = evaluate(genome, validation_episodes(config), seed, f"validation:q1007:v01:slot{slot}", treatment, repeats=config["repeats"])
                    rows[slot].update(validation=metrics, validation_records=records)
                    episode_calls += len(records)
                else:
                    metrics = rows[slot]["training"]
                candidate = {"genome": genome.to_dict(), "digest": genome.digest, "metrics": metrics,
                             "round": generation, "slot": slot,
                             "selection_source": "validation" if validation else "training_pilot"}
                if final_candidate is None or selection_key(metrics) > selection_key(final_candidate["metrics"]):
                    final_candidate = candidate
                if best is None or selection_key(metrics) > selection_key(best["metrics"]):
                    best = candidate
            if generation == 0:
                initial = final_candidate
        if str(generation + 1) in config["endpoint_rounds"]:
            label = config["endpoint_rounds"][str(generation + 1)]
            endpoints[label] = {"candidate": best, "through_round": generation + 1, "logical_episode_slots": episode_calls}
        history.append({"round": generation, "training_mean_reward": sum(r["training"]["mean_reward"] for r in rows) / size,
                        "training_mean_score": sum(scores) / size,
                        "training_mean_success": sum(r["training"]["success_rate"] for r in rows) / size,
                        "best_training_reward": max(r["training"]["mean_reward"] for r in rows), "best_training_score": max(scores), "validation_performed": validation,
                        "best_validation": best["metrics"] if best else None,
                        "mean_modules": sum(len(g.modules) for g in population) / size,
                        "ranking": {key: value for key, value in rank_diagnostic.items() if not key.endswith("order")}})
        attempt.update(status="evaluated", completed_episode_calls=episode_calls - calls_before,
                       completed_single_step_actions=sum(len(r["actions"]) for row in rows for field in ("training_records", "validation_records") for r in row.get(field, [])))
        write_json(attempt_path, attempt)
        if config["phase"] != "verification" and (generation == 0 or generation == rounds - 1 or (generation + 1) % 10 == 0):
            print(f"{condition}/{method}/{seed} round={generation + 1}/{rounds}", flush=True)
        name = f"round_{generation:03d}.json.gz"
        write_once(path / name, {"round": generation, "training_episodes": [e.episode_id for e in sampled], "candidates": rows,
                                 "ranking_diagnostics": rank_diagnostic})
        hashes[name] = file_digest(path / name)
        if generation < rounds - 1:
            if method == "evolution":
                population, ancestry = breed(population, [row["fitness_key"] for row in rows], rng, allocator, "F0")
            else:
                population = [initial_genome(rng, settings) for _ in range(size)]
                ancestry = [{"kind": "independent_initialization", "parents": [], "mutation": None} for _ in population]
        state = {"identity": identity, "next_round": generation + 1,
                 "population": [g.export() for g in population], "ancestry": ancestry,
                 "rng_state": rng.getstate(), "next_module_id": allocator.next_id,
                 "endpoints": endpoints, "best": best, "initial": initial, "initial_population": initial_population, "history": history,
                 "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                 "logical_episode_slots": episode_calls, "record_hashes": hashes, "result_digest": None,
                 "episode_accounting": attempt_accounting(path)}
        if generation == rounds - 1:
            result = {"identity": identity, "endpoints": endpoints, "best": best, "initial": initial, "final": final_candidate,
                      "initial_population": initial_population, "history": history,
                      "training_slots": size * rounds, "validation_slots": size * len(config["validation_rounds"]),
                      "logical_episode_slots": episode_calls}
            write_once(path / "result.json", result)
            state["result_digest"] = digest_json(result)
        state["checkpoint_digest"] = digest_json(state)
        write_json(checkpoint, state)
        attempt.update(status="committed", elapsed_seconds=time.perf_counter() - started)
        write_json(attempt_path, attempt)
        if pause_after is not None and generation + 1 >= pause_after:
            return {"paused_after": generation + 1}
    return result

