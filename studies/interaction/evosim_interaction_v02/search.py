"""조건별 탐색·원자적 회차 기록·엄격한 재개. 테스트는 여기서 열지 않는다."""

from pathlib import Path
import random
import time
from datetime import datetime, timezone
import uuid

from .datasets import make_split
from .evaluation import evaluate, selection_key
from .genetics import next_population
from .network import Genome, ModuleIds, initial_genome
from .protocol import check_access, load
from .rollout import derive_seed
from .settings import Settings, digest_json
from .storage import as_tuple, file_digest, read_json, write_json, write_once


def directory(root, condition, method, seed):
    return Path(root) / condition / f"{method}_{seed}"


def attempt_accounting(path):
    known, interrupted, upper = 0, 0, 0
    for file in (Path(path) / "attempts").glob("*.json"):
        item = read_json(file)
        if item["status"] in ("evaluated", "committed"):
            known += item["completed_episode_calls"]
        else:
            interrupted += 1
            upper += item["maximum_episode_calls"]
    return {"known_completed_episode_calls": known, "interrupted_attempts": interrupted,
            "unknown_interrupted_calls_upper_bound": upper}


def validate_checkpoint(state):
    body = {key: value for key, value in state.items() if key != "checkpoint_digest"}
    if state.get("checkpoint_digest") != digest_json(body):
        raise ValueError("Checkpoint checksum mismatch")


def run_search(root, condition, method, seed, *, pause_after=None):
    started = time.perf_counter()
    root = Path(root)
    config = load(root)
    if condition not in config["conditions"] or method not in config["methods"] or seed not in config["seeds"]:
        raise ValueError("Job outside protocol")
    settings = Settings.from_dict(config["settings"][condition])
    rounds, size = config["rounds"], config["population"]
    path = directory(root, condition, method, seed)
    identity = {"experiment_id": config["experiment_id"], "phase": config["phase"],
                "protocol_digest": config["protocol_digest"], "source_digest": config["source_digest"],
                "settings_digest": settings.digest, "datasets_digest": config["datasets_digest"],
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
        ancestry, history = state["ancestry"], state["history"]
        hashes, start, elapsed_before = state["record_hashes"], state["next_round"], state["elapsed_seconds"]
        episode_calls = state["logical_episode_slots"]
    else:
        initial_rng = random.Random(derive_seed("initial", seed))
        population = [initial_genome(initial_rng, settings) for _ in range(size)]
        initial_population = [g.to_dict() for g in population]
        rng = random.Random(derive_seed("search", seed, method))
        allocator = ModuleIds()
        best = initial = None
        ancestry = [{"kind": "initial", "parents": [], "mutation": None} for _ in population]
        history, hashes = [], {}
        start, elapsed_before, episode_calls = 0, 0.0, 0
    split = make_split()
    for generation in range(start, rounds):
        calls_before = episode_calls
        attempt_path = path / "attempts" / f"{generation:03d}_{uuid.uuid4().hex}.json"
        attempt = {"round": generation, "status": "started", "started_utc": datetime.now(timezone.utc).isoformat(),
                   "maximum_episode_calls": size * (24 + (252 * 3 if generation in config["validation_rounds"] else 0))}
        write_json(attempt_path, attempt)
        sampled = random.Random(derive_seed("training_batch", seed, generation)).sample(list(split.train), 24)
        check_access(config, f"train:{generation}", [e.episode_id for e in sampled])
        rows, scores = [], []
        for slot, genome in enumerate(population):
            metrics, records = evaluate(genome, sampled, seed, f"train:{generation}", settings)
            episode_calls += len(records)
            scores.append(metrics["mean_reward"])
            rows.append({"slot": slot, "genome": genome.to_dict(), "digest": genome.digest,
                         "ancestry": ancestry[slot], "training": metrics, "training_records": records})
        validation = generation in config["validation_rounds"]
        final_candidate = None
        if validation or config["phase"] not in ("main", "confirmation"):
            for slot, genome in enumerate(population):
                if validation:
                    check_access(config, "validation", [e.episode_id for e in split.validation])
                    metrics, records = evaluate(genome, split.validation, seed, "validation", settings, repeats=3)
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
        history.append({"round": generation, "training_mean_reward": sum(scores) / size,
                        "training_mean_success": sum(r["training"]["success_rate"] for r in rows) / size,
                        "best_training_reward": max(scores), "validation_performed": validation,
                        "best_validation": best["metrics"] if best else None,
                        "mean_modules": sum(len(g.modules) for g in population) / size})
        attempt.update(status="evaluated", completed_episode_calls=episode_calls - calls_before)
        write_json(attempt_path, attempt)
        if config["phase"] != "verification" and (generation == 0 or generation == rounds - 1 or (generation + 1) % 10 == 0):
            print(f"{condition}/{method}/{seed} round={generation + 1}/{rounds}", flush=True)
        name = f"round_{generation:03d}.json.gz"
        write_once(path / name, {"round": generation, "training_episodes": [e.episode_id for e in sampled], "candidates": rows})
        hashes[name] = file_digest(path / name)
        if generation < rounds - 1:
            if method == "evolution":
                population, ancestry = next_population(population, scores, rng, allocator, settings)
            else:
                population = [initial_genome(rng, settings) for _ in range(size)]
                ancestry = [{"kind": "independent_initialization", "parents": [], "mutation": None} for _ in population]
        state = {"identity": identity, "next_round": generation + 1,
                 "population": [g.export() for g in population], "ancestry": ancestry,
                 "rng_state": rng.getstate(), "next_module_id": allocator.next_id,
                 "best": best, "initial": initial, "initial_population": initial_population, "history": history,
                 "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                 "logical_episode_slots": episode_calls, "record_hashes": hashes, "result_digest": None,
                 "episode_accounting": attempt_accounting(path)}
        if generation == rounds - 1:
            result = {"identity": identity, "best": best, "initial": initial, "final": final_candidate,
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


def candidate_manifest(root, config, *, audit_rounds=True):
    manifest = {"protocol_digest": config["protocol_digest"], "jobs": {}}
    for condition in config["conditions"]:
        settings = Settings(condition)
        for method in config["methods"]:
            for seed in config["seeds"]:
                path = directory(root, condition, method, seed)
                if not (path / "checkpoint.json").exists() or not (path / "result.json").exists():
                    raise ValueError("Every protocol job must finish before candidate lock")
                state = read_json(path / "checkpoint.json")
                validate_checkpoint(state)
                result = read_json(path / "result.json")
                expected = {"condition": condition, "method": method, "seed": seed,
                            "protocol_digest": config["protocol_digest"], "settings_digest": settings.digest}
                if state["next_round"] != config["rounds"] or any(result["identity"][k] != v for k, v in expected.items()):
                    raise ValueError("Every protocol job must finish before candidate lock")
                if state["identity"] != result["identity"] or state["result_digest"] != digest_json(result):
                    raise ValueError("Checkpoint/result mismatch")
                if set(state["record_hashes"]) != {f"round_{g:03d}.json.gz" for g in range(config["rounds"])}:
                    raise ValueError("Completed round manifest incomplete")
                if audit_rounds:
                    for name, digest in state["record_hashes"].items():
                        if file_digest(path / name) != digest:
                            raise ValueError("Round changed after completion")
                roles = {}
                for role in ("best", "initial", "final"):
                    candidate = result[role]
                    genome = Genome.from_dict(candidate["genome"], settings)
                    if genome.digest != candidate["digest"]:
                        raise ValueError("Candidate genome digest mismatch")
                    roles[role] = {key: candidate[key] for key in ("digest", "round", "slot", "metrics", "selection_source")}
                manifest["jobs"][f"{condition}/{method}_{seed}"] = {
                    "roles": roles, "result_digest": digest_json(result), "record_hashes": state["record_hashes"]}
    return manifest


def lock_candidates(root, blind_selection):
    root = Path(root)
    config = load(root)
    if config["phase"] not in ("main", "confirmation"):
        raise ValueError("Pilot/verification cannot expose test or lock test candidates")
    from .finalize import validate_blind_selection
    validate_blind_selection(root, blind_selection)
    manifest = candidate_manifest(root, config)
    manifest["blind"] = blind_selection
    manifest["lock_digest"] = digest_json(manifest)
    write_once(root / "candidate_lock.json", manifest)
    return manifest


def verify_candidate_lock(root, config=None, *, audit_rounds=True):
    root = Path(root)
    config = config or load(root)
    locked = read_json(root / "candidate_lock.json")
    body = {key: value for key, value in locked.items() if key != "lock_digest"}
    if config["phase"] not in ("main", "confirmation") or digest_json(body) != locked["lock_digest"]:
        raise ValueError("Candidate lock invalid")
    current = candidate_manifest(root, config, audit_rounds=audit_rounds)
    if current["jobs"] != locked["jobs"] or current["protocol_digest"] != locked["protocol_digest"]:
        raise ValueError("Candidates changed after lock")
    if file_digest(root / "blind_validation.json.gz") != locked["blind"]["validation_artifact_sha256"]:
        raise ValueError("Blind validation changed after lock")
    return locked
