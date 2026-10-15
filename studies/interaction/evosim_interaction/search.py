"""Matched evolutionary and independent-random search, with round checkpoints."""

import random
import time
from pathlib import Path

from .datasets import make_split
from .evaluation import evaluate, selection_key
from .network import Genome, ModuleIds, SETTINGS, crossover, initial_genome, mutate
from .rollout import derive_seed
from .storage import as_tuple, read_json, write_json


def next_population(population, scores, rng, allocator):
    ranked = sorted(range(len(population)), key=lambda idx: (-scores[idx], idx))
    children, ancestry = [], []
    elites = max(1, int(len(population) * SETTINGS["elite_ratio"]))
    for idx in ranked[:elites]:
        children.append(population[idx].clone())
        ancestry.append({"kind": "elite", "parents": [idx], "mutation": None})

    def select():
        slots = rng.sample(range(len(population)), min(SETTINGS["tournament_size"], len(population)))
        return max(slots, key=lambda idx: (scores[idx], -idx))

    while len(children) < len(population):
        first = select()
        parents = [first]
        if rng.random() < SETTINGS["crossover_rate"]:
            second = select()
            parents.append(second)
            child = crossover(population[first], population[second], rng,
                              scores[first], scores[second], first, second)
            kind = "crossover"
        else:
            child = population[first].clone()
            kind = "clone"
        mutation = mutate(child, rng, allocator) if rng.random() < SETTINGS["mutation_rate"] else None
        children.append(child)
        ancestry.append({"kind": kind, "parents": parents, "mutation": mutation})
    return children, ancestry


def run_search(root, seed, method, rounds=100, size=20, pilot=False, pause_after=None):
    if method not in ("evolution", "random"):
        raise ValueError(method)
    started = time.perf_counter()
    root = Path(root)
    directory = root / f"{method}_{seed}"
    checkpoint_path = directory / "checkpoint.json"
    protocol = read_json(root / "protocol.json")
    identity = {"seed": seed, "method": method, "rounds": rounds, "size": size,
                "pilot": pilot, "source_digest": protocol["source_digest"]}
    if checkpoint_path.exists():
        state = read_json(checkpoint_path)
        if state["identity"] != identity:
            raise ValueError("Resume configuration differs")
        if state["next_round"] == rounds:
            return read_json(directory / "result.json")
        population = [Genome.from_dict(item) for item in state["population"]]
        rng = random.Random()
        rng.setstate(as_tuple(state["rng_state"]))
        allocator = ModuleIds(state["next_module_id"])
        ancestry = state["ancestry"]
        best = state["best"]
        initial = state["initial"]
        initial_population = state["initial_population"]
        history = state["history"]
        start_round = state["next_round"]
        elapsed_before = state["elapsed_seconds"]
    else:
        initial_rng = random.Random(derive_seed("initial", seed))
        population = [initial_genome(initial_rng) for _ in range(size)]
        initial_population = [genome.to_dict() for genome in population]
        rng = random.Random(derive_seed("search", seed, method))
        allocator = ModuleIds()
        ancestry = [{"kind": "initial", "parents": [], "mutation": None} for _ in population]
        best = initial = None
        history = []
        start_round = 0
        elapsed_before = 0.0
    split = make_split()
    for generation in range(start_round, rounds):
        sampled = random.Random(derive_seed("training_batch", seed, generation)).sample(list(split.train), 24)
        rows, scores = [], []
        for slot, genome in enumerate(population):
            metrics, records = evaluate(genome, sampled, seed, f"train:{generation}")
            scores.append(metrics["mean_reward"])
            rows.append({"slot": slot, "genome": genome.to_dict(), "digest": genome.digest,
                         "ancestry": ancestry[slot], "training": metrics, "training_records": records})
        validation_round = not pilot and (generation == 0 or generation % 10 == 9)
        final_candidate = None
        if validation_round or pilot:
            for slot, genome in enumerate(population):
                if pilot:
                    metrics = rows[slot]["training"]
                else:
                    metrics, records = evaluate(genome, split.validation, seed, "validation", repeats=3)
                    rows[slot]["validation"] = metrics
                    rows[slot]["validation_records"] = records
                candidate = {"genome": genome.to_dict(), "digest": genome.digest, "metrics": metrics,
                             "round": generation, "slot": slot,
                             "selection_source": "training_pilot" if pilot else "validation"}
                if final_candidate is None or selection_key(metrics) > selection_key(final_candidate["metrics"]):
                    final_candidate = candidate
                if best is None or selection_key(metrics) > selection_key(best["metrics"]):
                    best = candidate
            if generation == 0:
                initial = final_candidate
        history.append({"round": generation, "training_mean_reward": sum(scores) / size,
                        "training_mean_success": sum(r["training"]["success_rate"] for r in rows) / size,
                        "best_training_reward": max(scores),
                        "validation_performed": validation_round,
                        "best_validation": best["metrics"] if best else None,
                        "mean_modules": sum(len(g.modules) for g in population) / size})
        write_json(directory / f"round_{generation:03d}.json.gz", {
            "round": generation, "training_episodes": [e.episode_id for e in sampled], "candidates": rows,
        })
        if generation < rounds - 1:
            if method == "evolution":
                population, ancestry = next_population(population, scores, rng, allocator)
            else:
                population = [initial_genome(rng) for _ in range(size)]
                ancestry = [{"kind": "independent_initialization", "parents": [], "mutation": None}
                            for _ in population]
        elapsed = elapsed_before + time.perf_counter() - started
        state = {"identity": identity, "next_round": generation + 1,
                 "population": [g.to_dict() for g in population], "ancestry": ancestry,
                 "rng_state": rng.getstate(), "next_module_id": allocator.next_id,
                 "initial_population": initial_population, "initial": initial,
                 "best": best, "history": history, "elapsed_seconds": elapsed}
        # The result is committed first so a completed checkpoint always has a result.
        if generation == rounds - 1:
            result = {"identity": identity, "best": best, "initial": initial,
                      "final": final_candidate, "initial_population": initial_population,
                      "history": history, "elapsed_seconds": elapsed,
                      "training_slots": size * rounds,
                      "validation_slots": size * sum(h["validation_performed"] for h in history)}
            write_json(directory / "result.json", result)
        write_json(checkpoint_path, state)
        if (generation + 1) % 10 == 0 or generation == 0:
            print(f"{method} seed={seed} round={generation + 1}/{rounds} reward={history[-1]['training_mean_reward']:.4f}", flush=True)
        if pause_after is not None and generation + 1 >= pause_after:
            return {"paused_after": generation + 1}
    return result
