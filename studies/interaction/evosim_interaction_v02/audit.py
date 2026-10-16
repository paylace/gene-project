"""저장된 모든 개체·점수·후보 선발·유전 연산을 재구성한다."""

import random

from .datasets import make_split
from .evaluation import summarize, selection_key
from .genetics import next_population
from .network import Genome, ModuleIds, initial_genome
from .protocol import load
from .rollout import derive_seed
from .search import directory, run_search
from .settings import Settings
from .storage import as_tuple, read_json


def check_metrics(metrics, records):
    calculated = summarize(records)
    if any(metrics[key] != value for key, value in calculated.items()):
        raise ValueError("Recorded metrics disagree with episode records")
    for row in records:
        if not 1 <= len(row["actions"]) <= 24 or len(row["rewards_cents"]) != len(row["actions"]):
            raise ValueError("Invalid trajectory length")
        if set(row["actions"]) - set("012") or any(value not in (-1, 99) for value in row["rewards_cents"]):
            raise ValueError("Invalid action/reward")
        if row["collected"] != sum(value == 99 for value in row["rewards_cents"]) or row["success"] != (row["collected"] == 3):
            raise ValueError("Collected/success count mismatch")
        if not row["success"] and len(row["actions"]) != 24:
            raise ValueError("Failed episode must exhaust action budget")


def audit_job(root, condition, method, seed):
    config = load(root)
    path = directory(root, condition, method, seed)
    state = read_json(path / "checkpoint.json")
    if state["next_round"] != config["rounds"]:
        raise ValueError("Audit requires a completed search; it never runs missing rounds")
    result = run_search(root, condition, method, seed)  # 완료 기록 검증만 허용
    settings = Settings(condition)
    split = make_split()
    init_rng = random.Random(derive_seed("initial", seed))
    expected = [initial_genome(init_rng, settings) for _ in range(config["population"])]
    if [g.to_dict() for g in expected] != result["initial_population"]:
        raise ValueError("Initial population differs")
    rng = random.Random(derive_seed("search", seed, method))
    allocator = ModuleIds()
    expected_ancestry = [{"kind": "initial", "parents": [], "mutation": None} for _ in expected]
    best = initial = final = None
    checked, episode_slots = 0, 0
    for generation in range(config["rounds"]):
        saved = read_json(path / f"round_{generation:03d}.json.gz")
        sampled = random.Random(derive_seed("training_batch", seed, generation)).sample(list(split.train), 24)
        if saved["training_episodes"] != [e.episode_id for e in sampled]:
            raise ValueError("Training batch differs")
        rows = saved["candidates"]
        if [row["slot"] for row in rows] != list(range(config["population"])):
            raise ValueError("Candidate slots incomplete")
        population, scores = [], []
        round_best = None
        for row, expected_genome, ancestry in zip(rows, expected, expected_ancestry):
            genome = Genome.from_dict(row["genome"], settings)
            if genome.digest != row["digest"] or genome.digest != expected_genome.digest:
                raise ValueError("Population differs from replayed genetic operations")
            if row["ancestry"] != ancestry:
                raise ValueError("Ancestry differs from replayed genetic operations")
            population.append(genome)
            check_metrics(row["training"], row["training_records"])
            if [(r["episode_id"], r["repeat"]) for r in row["training_records"]] != [(e.episode_id, 0) for e in sampled]:
                raise ValueError("Training record split/order mismatch")
            scores.append(row["training"]["mean_reward"])
            episode_slots += len(row["training_records"])
            if generation in config["validation_rounds"]:
                check_metrics(row["validation"], row["validation_records"])
                expected_rows = [(e.episode_id, rep) for rep in range(3) for e in split.validation]
                if [(r["episode_id"], r["repeat"]) for r in row["validation_records"]] != expected_rows:
                    raise ValueError("Validation record split/order mismatch")
                episode_slots += len(row["validation_records"])
                metrics, source = row["validation"], "validation"
            elif config["phase"] not in ("main", "confirmation"):
                metrics, source = row["training"], "training_pilot"
            else:
                continue
            candidate = {"genome": genome.to_dict(), "digest": genome.digest, "metrics": metrics,
                         "round": generation, "slot": row["slot"], "selection_source": source}
            if round_best is None or selection_key(metrics) > selection_key(round_best["metrics"]):
                round_best = candidate
            if best is None or selection_key(metrics) > selection_key(best["metrics"]):
                best = candidate
        checked += len(rows)
        expected_history = {"round": generation, "training_mean_reward": sum(scores) / len(rows),
                            "training_mean_success": sum(r["training"]["success_rate"] for r in rows) / len(rows),
                            "best_training_reward": max(scores),
                            "validation_performed": generation in config["validation_rounds"],
                            "best_validation": best["metrics"] if best else None,
                            "mean_modules": sum(len(g.modules) for g in population) / len(rows)}
        if result["history"][generation] != expected_history:
            raise ValueError("History differs from audited records")
        if generation == 0:
            initial = round_best
        if generation == config["rounds"] - 1:
            final = round_best
        if generation < config["rounds"] - 1:
            if method == "evolution":
                expected, expected_ancestry = next_population(population, scores, rng, allocator, settings)
            else:
                expected = [initial_genome(rng, settings) for _ in population]
                expected_ancestry = [{"kind": "independent_initialization", "parents": [], "mutation": None} for _ in population]
    if any(result[role] != value for role, value in (("best", best), ("initial", initial), ("final", final))):
        raise ValueError("Candidate selection differs from full audit")
    if episode_slots != result["logical_episode_slots"] or rng.getstate() != as_tuple(state["rng_state"]) or allocator.next_id != state["next_module_id"]:
        raise ValueError("Final counters/RNG/id allocator disagree")
    if [Genome.restore(item, settings).to_dict() for item in state["population"]] != [g.to_dict() for g in population]:
        raise ValueError("Checkpoint population differs from final round")
    return {"passed": True, "condition": condition, "method": method, "seed": seed,
            "genomes_checked": checked, "episode_slots_checked": episode_slots}


def audit_test_job(root, condition, method, seed):
    from .environment import LineCollectEnv
    from .search import verify_candidate_lock
    config = load(root)
    locked = verify_candidate_lock(root, config, audit_rounds=False)
    path = directory(root, condition, method, seed)
    tested = read_json(path / "test.json.gz")
    lookup = {e.episode_id: e for e in make_split().test}
    expected_ids = [(e.episode_id, rep) for rep in range(3) for e in make_split().test]
    if tested["lock_digest"] != locked["lock_digest"]:
        raise ValueError("Test candidate lock differs")
    for role, item in tested["roles"].items():
        if item["digest"] != locked["jobs"][f"{condition}/{method}_{seed}"]["roles"][role]["digest"]:
            raise ValueError("Tested genome differs from locked candidate")
        check_metrics(item["metrics"], item["records"])
        if [(r["episode_id"], r["repeat"]) for r in item["records"]] != expected_ids:
            raise ValueError("Test records incomplete")
        for row in item["records"]:
            env, walls, failed = LineCollectEnv(), 0, 0
            env.reset(lookup[row["episode_id"]])
            for action, reward in zip(row["actions"], row["rewards_cents"]):
                result = env.step(int(action))
                if round(result.reward * 100) != reward:
                    raise ValueError("Test reward differs from environment replay")
                walls += result.wall_collision
                failed += result.failed_collect
            if not env.state.terminated or env.state.collected != row["collected"] or walls != row["walls"] or failed != row["failed_collect"]:
                raise ValueError("Test outcome differs from environment replay")
    return {"passed": True, "condition": condition, "method": method, "seed": seed,
            "roles": 3, "episode_records_replayed": len(expected_ids) * 3}
