"""단계별 자료 접근과 실험·의존 소스·환경의 고정."""

from pathlib import Path
import platform
import sys
import numpy as np

from evosim_interaction_v02.datasets import make_split
from evosim_interaction_v02.settings import digest_json
from evosim_interaction_v02.storage import read_json, write_once, file_digest, manifest_digest
from .source import ROOT, source_manifest, plan
from .model import CONDITIONS, Treatment

RESULTS = ROOT / "studies/interaction/results_q1005_v01"


def jobs_for(conditions, seeds, *, all_random=False):
    jobs = [{"condition": c, "method": "evolution", "seed": s} for c in conditions for s in seeds]
    jobs += [{"condition": c, "method": "random", "seed": s}
             for c in (conditions if all_random else ("R0S0",)) for s in seeds]
    return jobs


def validate_plan(value):
    exact = {"network": "A0", "lifetime_learning": False, "random_control": "R0S0", "pilot_rounds": 5,
             "validation_rounds": [0] + list(range(9, 100, 10)),
             "reward": {"progress_unit_denominator": 60, "maximum_units_per_goal": 6,
                        "maximum_units_per_episode": 18, "eta": 0.1},
             "statistics": {"bootstrap_seed": 72005, "bootstrap_draws": 10000, "alpha": 0.05,
                            "minimum_mean_difference": 0.1, "minimum_positive_seeds": 15, "sample_size": 20,
                            "ci": "individual percentile intervals, not simultaneous", "numeric_boundary_tolerance": 1e-12},
             "budgets": {"pilot_jobs": 10, "pilot_training_episodes": 24000, "main_jobs": 100,
                         "main_search_episodes": 21432000, "main_test_slots": 453600,
                         "confirmation_jobs_max": 60, "confirmation_search_episodes_max": 12859200,
                         "confirmation_test_slots_max": 272160}}
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ValueError("Numerical design is unsupported by this fixed implementation")
    if (value["experiment_id"] != "Q1-005" or value["conditions"] != list(CONDITIONS)
            or value["population"] != 20 or value["rounds"] != 100 or value["training_batch"] != 24
            or value["repeats"] != 3 or value["main_seeds"] != list(range(401, 421))
            or value["confirmation_seeds"] != list(range(501, 521)) or value["pilot_seeds"] != [911, 912]
            or value["reward"]["progress_unit_denominator"] != 60
            or value["statistics"]["sample_size"] != 20
            or value["primary_contrasts"] != ["R1S0", "R0S1", "R1S1"]
            or value["confirmation_priority"] != value["primary_contrasts"]):
        raise ValueError("Registered design differs")
    if make_split().digest != value["dataset_digest"]:
        raise ValueError("Registered datasets differ")


def prepare(root, phase, *, gate=None, pilot_report=None, verification_rounds=5, verification_size=4,
            verification_conditions=None, verification_methods=None, verification_validation=False):
    root = Path(root).resolve()
    spec = plan()
    validate_plan(spec)
    sources = source_manifest()
    source_digest = manifest_digest(sources)
    conditions = list(CONDITIONS)
    if phase == "verification":
        rounds, size, seeds = verification_rounds, verification_size, [41]
        if not 1 <= rounds <= 5 or not 2 <= size <= 20:
            raise ValueError("Verification budget out of range")
        conditions = verification_conditions or conditions
        if any(c not in CONDITIONS for c in conditions):
            raise ValueError("Unknown fixture condition")
        jobs = jobs_for(conditions, seeds, all_random=True)
        if verification_methods:
            jobs = [job for job in jobs if job["method"] in verification_methods]
        validations = sorted({0, rounds - 1}) if verification_validation else []
        if not jobs:
            raise ValueError("Empty verification jobs")
    elif phase in ("pilot", "main", "confirmation"):
        if root != (RESULTS / phase).resolve():
            raise ValueError("Official phase must use its registered new directory")
        evidence = read_json(gate) if gate else {}
        if (not evidence.get("passed") or not evidence.get("full_gate") or evidence.get("source_digest") != source_digest
                or evidence.get("plan_digest") != digest_json(spec)):
            raise ValueError("Passing matching-source/design gate required")
        rounds, size = spec["rounds"], spec["population"]
        validations = spec["validation_rounds"]
        seeds = spec[phase + "_seeds"]
        if phase == "pilot":
            rounds, validations = spec["pilot_rounds"], []
        elif phase == "main":
            pilot = read_json(pilot_report) if pilot_report else {}
            if (not pilot.get("all_jobs_complete") or pilot.get("source_digest") != source_digest
                    or pilot.get("plan_digest") != digest_json(spec)):
                raise ValueError("Matching completed training-only pilot required")
        else:
            previous = load(RESULTS / "main")
            main_summary = read_json(RESULTS / "main/summary.json")
            from .analysis import primary_decision
            expected_decision = primary_decision({name: main_summary["rates"][name]["evolution"] for name in CONDITIONS})
            if main_summary["decision"] != expected_decision:
                raise ValueError("Main decision differs from registered analysis")
            selected = main_summary["decision"]["confirmation_condition"]
            if selected not in spec["confirmation_priority"] or previous["source_digest"] != source_digest:
                raise ValueError("No eligible matching-source confirmation condition")
            conditions = ["R0S0", selected]
        jobs = jobs_for(conditions, seeds)
    else:
        raise ValueError("Unknown experiment phase")
    config = {"schema": 1, "experiment_id": "Q1-005", "phase": phase, "locked": True,
              "fixture": phase == "verification", "plan": spec, "plan_digest": digest_json(spec),
              "conditions": conditions, "seeds": seeds, "jobs": jobs, "population": size,
              "rounds": rounds, "training_batch": 24, "validation_rounds": validations, "repeats": 3,
              "sources": sources, "source_digest": source_digest, "datasets_digest": make_split().digest,
              "settings": {c: Treatment(c).to_dict() for c in conditions},
              "python": platform.python_version(), "numpy": np.__version__, "platform": sys.platform}
    if phase != "verification":
        config.update(gate_path=Path(gate).resolve().relative_to(ROOT).as_posix(), gate_sha256=file_digest(gate))
    if phase == "confirmation":
        config.update(main_summary_sha256=file_digest(RESULTS / "main/summary.json"),
                      shared_blind_sha256=file_digest(RESULTS / "main/blind_validation.json.gz"),
                      shared_baseline_sha256=file_digest(RESULTS / "main/baseline_results.json"))
    config["protocol_digest"] = digest_json(config)
    write_once(root / "datasets.json", make_split().manifest())
    write_once(root / "protocol.json", config)
    return config


def load(root):
    root = Path(root)
    config = read_json(root / "protocol.json")
    if config["protocol_digest"] != digest_json({k: v for k, v in config.items() if k != "protocol_digest"}):
        raise ValueError("Protocol checksum differs")
    if config["source_digest"] != manifest_digest(source_manifest()) or config["sources"] != source_manifest():
        raise ValueError("Experiment or frozen dependency source changed")
    if config["plan_digest"] != digest_json(plan()) or config["plan_digest"] != digest_json(config["plan"]):
        raise ValueError("Registered numerical design changed")
    if config["datasets_digest"] != make_split().digest or read_json(root / "datasets.json") != make_split().manifest():
        raise ValueError("Dataset manifest changed")
    for name, setting in config["settings"].items():
        if Treatment.from_dict(setting) != Treatment(name):
            raise ValueError("Treatment changed")
    if not config["fixture"]:
        if (config["python"], config["numpy"], config["platform"]) != (platform.python_version(), np.__version__, sys.platform):
            raise ValueError("Execution runtime differs")
        if file_digest(ROOT / config["gate_path"]) != config["gate_sha256"]:
            raise ValueError("Original execution gate changed")
        if root.resolve() != (RESULTS / config["phase"]).resolve():
            raise ValueError("Official protocol was moved into another experiment path")
    return config


def check_access(config, purpose, episodes):
    ids = [ep.episode_id for ep in episodes]
    split = make_split()
    if purpose.startswith("train:"):
        allowed = {e.episode_id for e in split.train}
    elif purpose == "validation" and config["validation_rounds"]:
        allowed = {e.episode_id for e in split.validation}
    elif purpose == "test" and config["phase"] in ("main", "confirmation"):
        allowed = {e.episode_id for e in split.test}
    else:
        raise ValueError("Data purpose forbidden in phase")
    if not ids or not set(ids) <= allowed:
        raise ValueError("Episode outside allowed split")


def evaluate_split(root, condition, method, seed, genome, episodes, purpose, *, repeats=1):
    from .evaluation import evaluate
    config = load(root)
    job = {"condition": condition, "method": method, "seed": seed}
    if job not in config["jobs"]:
        raise ValueError("Evaluation job outside protocol")
    episodes = tuple(episodes)
    check_access(config, purpose, episodes)
    if purpose == "test":
        from .search import verify_candidate_lock
        locked = verify_candidate_lock(root, config, audit_rounds=False)
        roles = locked["jobs"][f"{condition}/{method}_{seed}"]["roles"]
        if genome.digest not in {role["digest"] for role in roles.values()}:
            raise ValueError("Test genome is not a locked candidate for this job")
        if episodes != make_split().test or repeats != 3:
            raise ValueError("Production test must use the complete registered test and repeats")
    return evaluate(genome, episodes, seed, purpose, Treatment(condition), repeats=repeats,
                    training=purpose.startswith("train:"))
