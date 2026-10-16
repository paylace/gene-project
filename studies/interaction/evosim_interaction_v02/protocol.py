"""설정·소스·자료를 잠그고 단계별 데이터 접근을 제한한다."""

from pathlib import Path
import platform
import sys

import numpy as np

from .datasets import make_split
from .settings import Settings, digest_json
from .storage import read_json, write_once, source_manifest, manifest_digest

ROOT = Path(__file__).resolve().parents[3]
PLAN = ROOT / "docs/protocols/q1_004_v02_draft.json"


def prepare(root, phase, *, gate=None, pilot_report=None, main_root=None, verification_rounds=5, verification_size=4):
    root = Path(root)
    plan = read_json(PLAN)
    sources = source_manifest()
    source_digest = manifest_digest(sources)
    if phase == "verification":
        seeds, conditions, rounds, size = [41], ["A0", "A1", "A2"], verification_rounds, verification_size
        if not 1 <= rounds <= 5 or not 2 <= size <= 20:
            raise ValueError("Verification budget out of range")
    elif phase in ("pilot", "main", "confirmation"):
        expected = ROOT / plan["paths"]["results"] / phase
        if root.resolve() != expected.resolve():
            raise ValueError("Official phase must use the planned new output directory")
        evidence = read_json(gate) if gate else {}
        if not evidence.get("passed") or evidence.get("source_digest") != source_digest:
            raise ValueError("Current source requires a passing W3-W6 gate")
        if phase == "main":
            cost = read_json(pilot_report) if pilot_report else {}
            if not cost.get("all_jobs_complete") or cost.get("source_digest") != source_digest:
                raise ValueError("Main requires the matching pilot cost/resume report")
        spec = plan["phases"][phase]
        seeds, conditions, rounds, size = spec["seeds"], spec["conditions"], spec["rounds"], plan["search"]["population"]
        if phase == "confirmation":
            parent = Path(main_root) if main_root else ROOT / plan["paths"]["results"] / "main"
            if parent.resolve() != (ROOT / plan["paths"]["results"] / "main").resolve():
                raise ValueError("Confirmation must reference the planned main experiment")
            previous = load(parent)
            summary = read_json(parent / "summary.json")
            selected = summary["decision"]["confirmation_condition"]
            if selected not in ("A1", "A2") or previous["source_digest"] != source_digest:
                raise ValueError("No eligible matching-source confirmation condition")
            conditions = ["A0", selected]
            parent = parent.resolve()
    else:
        raise ValueError("Unsupported phase; confirmation requires its separately resolved protocol")
    split = make_split()
    if split.digest != plan["dataset"]["digest"]:
        raise ValueError("Dataset differs from plan")
    config = {"schema": 1, "experiment_id": "Q1-004", "version": "q1-v0.2", "phase": phase,
              "locked": True, "kind": "verification_fixture_protocol" if phase == "verification" else "execution_protocol",
              "plan_digest": digest_json(plan), "plan": plan,
              "seeds": seeds, "conditions": conditions, "methods": ["evolution", "random"],
              "rounds": rounds, "population": size, "training_batch": 24,
              "validation_rounds": plan["search"]["validation_rounds"] if phase in ("main", "confirmation") else [],
              "evaluation_repeats": 3, "datasets_digest": split.digest,
              "settings": {name: Settings(name).to_dict() for name in conditions},
              "sources": sources, "source_digest": source_digest,
              "python": platform.python_version(), "numpy": np.__version__, "platform": sys.platform}
    if phase == "confirmation":
        from .storage import file_digest
        config.update(shared_baseline_root=parent.relative_to(ROOT).as_posix(),
                      shared_baseline_protocol=previous["protocol_digest"],
                      main_summary_sha256=file_digest(parent / "summary.json"),
                      shared_blind_sha256=file_digest(parent / "blind_validation.json.gz"))
    config["protocol_digest"] = digest_json(config)
    write_once(root / "datasets.json", split.manifest())
    write_once(root / "protocol.json", config)
    return config


def load(root):
    root = Path(root)
    config = read_json(root / "protocol.json")
    without_digest = {key: value for key, value in config.items() if key != "protocol_digest"}
    if config["protocol_digest"] != digest_json(without_digest):
        raise ValueError("Protocol digest mismatch")
    if config["source_digest"] != manifest_digest(source_manifest()):
        raise ValueError("Source changed after protocol lock")
    if config["phase"] != "verification" and (config["python"] != platform.python_version() or config["numpy"] != np.__version__ or config["platform"] != sys.platform):
        raise ValueError("Runtime differs from the locked execution environment")
    if config["datasets_digest"] != make_split().digest or read_json(root / "datasets.json") != make_split().manifest():
        raise ValueError("Datasets changed after protocol lock")
    if config["plan_digest"] != digest_json(config["plan"]):
        raise ValueError("Plan digest mismatch")
    for name, data in config["settings"].items():
        if Settings.from_dict(data) != Settings(name):
            raise ValueError("Settings mismatch")
    return config


def check_access(config, purpose, episode_ids):
    split = make_split()
    if purpose.startswith("train:"):
        allowed = {e.episode_id for e in split.train}
    elif purpose == "validation" and config["phase"] in ("main", "confirmation"):
        allowed = {e.episode_id for e in split.validation}
    elif purpose == "test" and config["phase"] in ("main", "confirmation"):
        allowed = {e.episode_id for e in split.test}
    else:
        raise ValueError("Data purpose is forbidden in this phase")
    if not episode_ids or not set(episode_ids) <= allowed:
        raise ValueError("Episodes outside the permitted split")


def evaluate_split(root, genome, episodes, seed, purpose, *, repeats=1, observe=False):
    from .evaluation import evaluate
    root = Path(root)
    config = load(root)
    episodes = tuple(episodes)
    check_access(config, purpose, [e.episode_id for e in episodes])
    if purpose == "test":
        from .search import verify_candidate_lock
        locked = verify_candidate_lock(root, config, audit_rounds=False)
        allowed = [role["digest"] for key, job in locked["jobs"].items()
                   if key.startswith(genome.settings.condition + "/") and key.endswith("_" + str(seed))
                   for role in job["roles"].values()]
        if genome.digest not in allowed:
            raise ValueError("Genome is not a locked candidate for this condition/seed")
    return evaluate(genome, episodes, seed, purpose, genome.settings, repeats=repeats, observe=observe)
